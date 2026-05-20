from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Dict

import redis.asyncio as aioredis

from backend.core.config import  get_config
from backend.core.logger import  get_logger
from .generator_base import AnswerGeneratorBase
from .groq_generator import get_groq_generator, GroqRateLimitError
from .groq_generator import get_groq_generator
from .hf_generator import get_hf_generator

logger = get_logger(__name__)

REDIS_URL = get_config().REDIS_URL
REDIS_KEY = "groq:daily_request_count"
REDIS_RESET_KEY = "groq:last_reset_date"


class SmartGeneratorGuard(AnswerGeneratorBase):
    def __init__(
        self,
        groq_daily_limit: int = get_config().GENERATOR_GROQ_DAILY_LIMIT,
        soft_threshold_pct: float = 0.8,
        hard_threshold_pct: float = 1.0,
    ):
        super().__init__()

        self._groq = get_groq_generator()
        self._hf = get_hf_generator()

        self._groq_daily_limit = groq_daily_limit
        self._soft_threshold = int(groq_daily_limit * soft_threshold_pct)
        self._hard_threshold = groq_daily_limit
        self._round_robin_toggle: bool = False

        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    def _get_utc_today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _check_daily_reset(self) -> None:
        today = self._get_utc_today_str()
        last_reset = await self._redis.get(REDIS_RESET_KEY)

        if last_reset != today:
            logger.info("Daily counter reset (UTC midnight).")
            await self._redis.set(REDIS_KEY, 0)
            await self._redis.set(REDIS_RESET_KEY, today)

    async def _get_request_count(self) -> int:
        await self._check_daily_reset()
        count = await self._redis.get(REDIS_KEY)
        return int(count) if count else 0

    async def _record_groq_request(self) -> None:
        count = await self._redis.incr(REDIS_KEY)
        remaining = self._groq_daily_limit - count
        logger.debug(f"Groq requests: {count}/{self._groq_daily_limit} (remaining: {remaining})")

    async def _get_strategy(self) -> str:
        count = await self._get_request_count()

        if count >= self._hard_threshold:
            logger.warning(f"Groq hard limit reached ({count}/{self._groq_daily_limit}). Switching to HF.")
            return "hf"

        if count >= self._soft_threshold:
            self._round_robin_toggle = not self._round_robin_toggle
            return "groq" if self._round_robin_toggle else "hf"

        return "groq"

    async def aclose(self) -> None:
        await self._groq.aclose()
        await self._hf.aclose()
        await self._redis.aclose()

    async def generate(
        self,
        question: str,
        context: str,
        sources: list[dict] | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        strategy = await self._get_strategy()
        logger.info(f"Generator strategy: {strategy}")

        try:
            if strategy == "groq":
                answer = await self._groq.generate(question, context, sources, history)
                await self._record_groq_request()
                return answer
            else:
                return await self._hf.generate(question, context, sources, history)

        except GroqRateLimitError as rate_exc:
            # Groq rate limited — switch to HF immediately, no waiting
            logger.warning(f"Groq rate limited ({rate_exc}). Falling back to HF immediately.")
            try:
                return await self._hf.generate(question, context, sources, history)
            except Exception as hf_exc:
                logger.error(f"HF fallback also failed: {hf_exc}")
                return "Sorry, I'm having trouble generating an answer right now. Please try again later."

        except Exception as primary_exc:
            logger.warning(f"{strategy.upper()} failed: {primary_exc}. Trying fallback...")

            fallback = self._hf if strategy == "groq" else self._groq
            try:
                answer = await fallback.generate(question, context, sources, history)
                if strategy == "groq":
                    await self._record_groq_request()
                return answer
            except Exception as fallback_exc:
                logger.error(f"Fallback also failed: {fallback_exc}")
                return "Sorry, I'm having trouble generating an answer right now. Please try again later."


@lru_cache(maxsize=1)
def get_smart_guard(
    groq_daily_limit: int = get_config().GENERATOR_GROQ_DAILY_LIMIT,
    soft_threshold_pct: float = 0.8,
    hard_threshold_pct: float = 1.0,
) -> SmartGeneratorGuard:
    return SmartGeneratorGuard(
        groq_daily_limit=groq_daily_limit,
        soft_threshold_pct=soft_threshold_pct,
        hard_threshold_pct=hard_threshold_pct,
    )

def reset_smart_guard_cache() -> None:
    get_smart_guard.cache_clear()