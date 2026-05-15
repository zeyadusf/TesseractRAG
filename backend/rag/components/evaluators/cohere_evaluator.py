from typing import List, Dict, Any, Optional
import json
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from .base_evaluator import BaseEvaluator
import cohere
from backend.core import get_config, get_logger

logger = get_logger(__name__)


class CohereRAGEvaluator(BaseEvaluator):
    """
    Lightweight RAG evaluator using Cohere API.
    Returns RAGAS-like metrics scaled to 0-100% with continuous decimals.
    """
    def __init__(self):
        super().__init__()
        self._co: Optional[cohere.AsyncClient] = None

    @staticmethod
    def _to_percentage(value: Optional[float], strict: bool = True) -> float:
        """Safely clamp & scale. In strict mode, penalize suspicious '1.0' scores."""
        if value is None:
            return 0.0
        v = float(value)
        
        if 0.0 <= v <= 1.0:
            v *= 100.0
        
        if strict and v == 100.0:
            v = 99.5  
        
        return round(max(0.0, min(100.0, v)), 1)

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response, with fallbacks."""
        try:
            clean = re.sub(r'```(?:json)?\s*|\s*```', '', text).strip()
            match = re.search(r'\{[\s\S]*\}', clean)
            if match:
                clean = match.group()
            return json.loads(clean)
        except (json.JSONDecodeError, re.error) as e:
            logger.warning(f"JSON parse failed: {e}. Raw: {text[:200]}")
            return {}

    async def _ensure_client(self) -> cohere.AsyncClient:
        """Lazy initialization & connection reuse."""
        if self._co is None:
            api_key = getattr(self, '_settings', None).COHERE_API_KEY if hasattr(self, '_settings') else None
            if not api_key:
                raise ValueError("COHERE_API_KEY not found in settings.")
            self._co = cohere.AsyncClient(api_key=api_key)
        return self._co

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_cohere_async(self, co: cohere.AsyncClient, prompt: str, model: str) -> str:
        response = await co.chat(
            message=prompt,
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.text

    async def evaluate(
        self,
        query: str,
        response_text: str,
        contexts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        co = await self._ensure_client()
        model = model or getattr(self._settings, 'COHERE_EVAL_MODEL', None) or "command-r-plus-08-2024"
        context_str = "\n\n---\n\n".join(contexts[:5])

        #  STRICT RUBRIC-BASED PROMPT
        prompt = f"""
You are a HYPER-CRITICAL RAG evaluator. Your job is to FIND FLAWS and assign PENALTIES.

[Query]: {query}
[Retrieved Context]:
{context_str}
[Assistant Response]: {response_text}

### STEP 1: ANALYSIS (Think before scoring)
1. List every claim in the response.
2. For each claim: is it DIRECTLY supported by context? Mark [✓] or [✗].
3. Does the response answer EVERY part of the query? If NO, what's missing?
4. Are ANY context chunks irrelevant or noisy?

### STEP 2: SCORING (Use the Rubric)
| Score Range | Meaning |
|-------------|---------|
| 95-100 | Perfect. Zero flaws. (Extremely rare) |
| 85-94 | Excellent. 1 minor issue. |
| 70-84 | Good. Missing 1 sub-point OR slight hallucination. |
| 50-69 | Partial. Missed major part of query OR multiple weak claims. |
| 25-49 | Poor. Mostly hallucinated or irrelevant. |
| 0-24 | Useless. Contradicts context or empty. |

### FEW-SHOT EXAMPLES (Learn from these):
Q: "What is A and B?" → A: "A is..." [ignores B]
→ answer_relevancy: 55.0 (missed 50% of query)

Q: "Explain X" → A: "X is great [no source]"
→ faithfulness: 30.0 (claim unsupported)

Q: Context has 3 chunks, only 1 is relevant
→ context_precision: 33.3

### STEP 3: OUTPUT JSON ONLY
{{
  "faithfulness": <0.0-100.0>,
  "answer_relevancy": <0.0-100.0>,
  "context_precision": <0.0-100.0>,
  "context_recall": <0.0-100.0>,
  "reasoning": "<1 sentence: main penalty reason>"
}}

 WARNING: If you return 100.0 without finding ANY flaw, you failed your job.
 Use DECIMALS: 72.5, 88.0, 45.3 — NEVER just 0 or 100.
"""

        try:
            raw_response = await self._call_cohere_async(co, prompt, model)
            parsed = self._parse_json_response(raw_response)

            return {
                "faithfulness":      self._to_percentage(parsed.get("faithfulness")),
                "answer_relevancy":  self._to_percentage(parsed.get("answer_relevancy")),
                "context_precision": self._to_percentage(parsed.get("context_precision")),
                "context_recall":    self._to_percentage(parsed.get("context_recall")),
                "reasoning":         str(parsed.get("reasoning", ""))[:200],
                "eval_model":        model,
            }
        except Exception as e:
            logger.error(f"Cohere evaluation failed: {type(e).__name__}: {str(e)[:100]}")
            return {
                "faithfulness":      0.0,
                "answer_relevancy":  0.0,
                "context_precision": 0.0,
                "context_recall":    0.0,
                "reasoning":         f"Eval error: {type(e).__name__}",
                "eval_model":        model,
            }

    async def aclose(self):
        """Properly close the async client and free connection pool."""
        if self._co:
            # await self._co.aclose()  # co.aesxit
            self._co = None