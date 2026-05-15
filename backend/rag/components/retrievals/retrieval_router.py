from __future__ import annotations

import re

from backend.core import get_logger

logger = get_logger(__name__)

# Phrases that signal a conceptual / natural-language question
CONCEPTUAL_STARTERS: frozenset[str] = frozenset({
    "what is", "what are", "what was", "what were",
    "how does", "how do", "how is", "how are", "how can",
    "why is", "why are", "why does", "why do",
    "explain", "describe", "tell me", "summarise", "summarize",
    "give me an overview", "what does",
})

# Identifier pattern: all-caps abbreviation, version tag, or 3+ digit number
_IDENTIFIER_RE = re.compile(r"\b([A-Z]{2,}|v\d+|\d{3,})\b")

# Valid explicit strategies callers may pass
_VALID_STRATEGIES: frozenset[str] = frozenset({"hybrid", "semantic", "lexical"})


class RetrievalRouter:
    """
    Pure routing logic — returns a strategy string, performs no retrieval.

    Strategy selection order
    ------------------------
    1. Explicit override  — caller passes a valid strategy → use it as-is.
    2. Short + identifier — ≤ 3 tokens AND contains an identifier (e.g. "BERT v2")
                            → lexical  (exact-match lookups benefit from BM25).
    3. Long conceptual    — > 5 tokens AND starts with a conceptual phrase
                            → semantic (meaning-heavy queries need embeddings).
    4. Default            → hybrid    (catches everything else).

    Note: _IDENTIFIER_RE searches the *original* query (case-sensitive) so
    that genuine abbreviations like "BERT" are detected, but common lowercase
    words are not false-positived as identifiers.
    """

    def route(self, query: str, user_strategy: str = "auto") -> str:
        logger.info("Router selecting strategy…")

        # ── Rule 1: explicit override ─────────────────────────────────────────
        if user_strategy != "auto":
            if user_strategy not in _VALID_STRATEGIES:
                logger.warning(
                    f"Unknown user_strategy '{user_strategy}', falling back to 'hybrid'"
                )
                return "hybrid"
            logger.info(f"Router: explicit strategy '{user_strategy}'")
            return user_strategy

        normalised = query.strip().lower()
        tokens = normalised.split()

        # ── Rule 2: short query with identifier → lexical ─────────────────────
        # Search the *original* (un-lowercased) query so regex picks up
        # genuine ALL-CAPS abbreviations without false-positiving on
        # common lowercase words.
        has_identifier = bool(_IDENTIFIER_RE.search(query))
        if len(tokens) <= 3 and has_identifier:
            logger.info("Router: short + identifier → 'lexical'")
            return "lexical"

        # ── Rule 3: long conceptual question → semantic ───────────────────────
        starts_conceptual = any(normalised.startswith(s) for s in CONCEPTUAL_STARTERS)
        if len(tokens) > 5 and starts_conceptual:
            logger.info("Router: long conceptual query → 'semantic'")
            return "semantic"

        # ── Rule 4: default ───────────────────────────────────────────────────
        logger.info("Router: default → 'hybrid'")
        return "hybrid"