"""
Test script for QueryRewriteDispatcher + GroqQueryRewriter
Run: python -m backend.testing.query_rewrite_test
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

# ── adjust import path if needed ──────────────────────────────────────────────
from backend.rag.components.query_rewrite.groq_query_rewriter import get_groq_rewriter
from backend.rag.components.query_rewrite.query_dispatcher import QueryRewriteDispatcher

# ── test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "what is machine learning",
    "how does RAG work",
    "climate change effects",
    "python async programming",
    "ما هو الذكاء الاصطناعي",          # Arabic query
    "",                                  # edge case: empty
    "   ",                               # edge case: whitespace
]

# ── helpers ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg: str)   -> None: print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg: str) -> None: print(f"  {RED}❌ {msg}{RESET}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg: str) -> None: print(f"  {CYAN}ℹ️  {msg}{RESET}")
def header(msg: str) -> None:
    print(f"\n{BOLD}{'─'*55}")
    print(f"  {msg}")
    print(f"{'─'*55}{RESET}")


def assert_check(condition: bool, pass_msg: str, fail_msg: str) -> bool:
    if condition:
        ok(pass_msg)
    else:
        fail(fail_msg)
    return condition


# ── individual tests ──────────────────────────────────────────────────────────
async def test_singleton() -> bool:
    header("Test 1 — Singleton (lru_cache)")
    r1 = get_groq_rewriter()
    r2 = get_groq_rewriter()
    passed = assert_check(
        r1 is r2,
        "get_groq_rewriter() returns same instance",
        "get_groq_rewriter() returned different instances!",
    )
    d1 = QueryRewriteDispatcher("groq")
    d2 = QueryRewriteDispatcher("groq")
    passed &= assert_check(
        d1._instance is d2._instance,
        "Dispatcher shares same provider instance",
        "Dispatcher created different provider instances!",
    )
    return passed


async def test_edge_cases(dispatcher: QueryRewriteDispatcher) -> bool:
    header("Test 2 — Edge Cases (empty / whitespace)")
    passed = True

    for q in ["", "   "]:
        result = await dispatcher.rewrite(q)
        passed &= assert_check(
            result == q,
            f"Empty/whitespace query returned as-is: {repr(q)}",
            f"Expected {repr(q)}, got {repr(result)}",
        )
    return passed


async def test_rewrite(dispatcher: QueryRewriteDispatcher) -> bool:
    header("Test 3 — rewrite()")
    passed = True

    for query in TEST_QUERIES:
        if not query.strip():
            continue

        start = time.perf_counter()
        result = await dispatcher.rewrite(query)
        elapsed = time.perf_counter() - start

        is_str    = isinstance(result, str)
        not_empty = bool(result.strip())
        has_fallback = result == query   # fallback if API failed

        passed &= assert_check(is_str and not_empty,
            f"[{elapsed:.2f}s] '{query[:30]}' → '{result[:60]}'",
            f"Bad result for '{query}': {repr(result)}",
        )
        if has_fallback:
            warn(f"Fallback used for: '{query[:40]}'")

    return passed


async def test_expand(dispatcher: QueryRewriteDispatcher) -> bool:
    header("Test 4 — expand()")
    passed = True

    for query in TEST_QUERIES:
        if not query.strip():
            continue

        start = time.perf_counter()
        result = await dispatcher.expand(query)
        elapsed = time.perf_counter() - start

        passed &= assert_check(
            isinstance(result, str) and bool(result.strip()),
            f"[{elapsed:.2f}s] '{query[:30]}' → '{result[:60]}'",
            f"Bad expand result for '{query}': {repr(result)}",
        )
    return passed


async def test_rewrite_and_expand(dispatcher: QueryRewriteDispatcher) -> bool:
    header("Test 5 — rewrite_and_expand() concurrent")
    passed = True
    query = "how does vector search work in RAG systems"

    start = time.perf_counter()
    result = await dispatcher.rewrite_and_expand(query)
    elapsed = time.perf_counter() - start

    expected_keys = {"original", "rewritten", "expanded"}
    passed &= assert_check(
        set(result.keys()) == expected_keys,
        f"Response has correct keys: {expected_keys}",
        f"Missing keys. Got: {set(result.keys())}",
    )
    passed &= assert_check(
        result["original"] == query,
        "original key matches input query",
        f"original mismatch: {repr(result.get('original'))}",
    )
    passed &= assert_check(
        bool(result["rewritten"].strip()),
        f"[{elapsed:.2f}s] rewritten: '{result['rewritten'][:60]}'",
        "rewritten is empty",
    )
    passed &= assert_check(
        bool(result["expanded"].strip()),
        f"[{elapsed:.2f}s] expanded:  '{result['expanded'][:60]}'",
        "expanded is empty",
    )
    info(f"Both ran concurrently in {elapsed:.2f}s total")
    return passed


async def test_concurrent_requests(dispatcher: QueryRewriteDispatcher) -> bool:
    header("Test 6 — Concurrent Requests (shared client)")
    queries = TEST_QUERIES[:4]  # skip empty ones implicitly

    start = time.perf_counter()
    results = await asyncio.gather(
        *[dispatcher.rewrite(q) for q in queries if q.strip()],
        return_exceptions=True,
    )
    elapsed = time.perf_counter() - start

    passed = True
    for q, r in zip([q for q in queries if q.strip()], results):
        if isinstance(r, Exception):
            fail(f"Exception for '{q}': {r}")
            passed = False
        else:
            ok(f"'{q[:30]}' → '{str(r)[:50]}'")

    info(f"All concurrent requests done in {elapsed:.2f}s")
    return passed


# ── main runner ───────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"\n{BOLD}{'═'*55}")
    print("  Query Rewrite — Full Test Suite")
    print(f"{'═'*55}{RESET}")

    dispatcher = QueryRewriteDispatcher(provider="groq")
    results: dict[str, bool] = {}

    try:
        results["Singleton"]            = await test_singleton()
        results["Edge Cases"]           = await test_edge_cases(dispatcher)
        results["rewrite()"]            = await test_rewrite(dispatcher)
        results["expand()"]             = await test_expand(dispatcher)
        results["rewrite_and_expand()"] = await test_rewrite_and_expand(dispatcher)
        results["Concurrent Requests"]  = await test_concurrent_requests(dispatcher)

    finally:
        await dispatcher.close()

    # ── summary ───────────────────────────────────────────────────────────────
    header("Summary")
    all_passed = True
    for name, passed in results.items():
        if passed:
            ok(name)
        else:
            fail(name)
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}  All tests passed ✅{RESET}\n")
    else:
        print(f"{RED}{BOLD}  Some tests failed ❌{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())