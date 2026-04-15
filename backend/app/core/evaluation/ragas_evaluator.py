from typing import List, Optional, Union, Dict, Any
import json
import re
import cohere 
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RagasEvaluator:

    @staticmethod
    def evaluate(query, response_text, contexts) -> Dict[str, Any]:
        co = cohere.Client(get_settings().COHERE_API_KEY)
        logger.info("Starting RAG evaluation...")
        context_str = "\n---\n".join(contexts)
        
        eval_prompt = f"""
        Evaluate this RAG response carefully.
        
        [Query]: {query}
        [Context]: {context_str}
        [Response]: {response_text}

        Return ONLY a valid JSON object with exactly these keys:
        {{
          "faithfulness": <float 0.0-1.0, is the response grounded in the context?>,
          "answer_relevancy": <float 0.0-1.0, does the response answer the query?>,
          "context_precision": <float 0.0-1.0, is the context relevant to the query?>,
          "reasoning": <short string explaining the scores>
        }}
        No markdown, no extra text, just the JSON.
        """

        try:
            eval_res = co.chat(
                message=eval_prompt,
                temperature=0.1
            )
            clean_text = re.sub(r'```json|```', '', eval_res.text).strip()
            result = json.loads(clean_text)
            
            # Ensure all required keys exist with fallback values
            normalized = {
                "faithfulness":      float(result.get("faithfulness", 0.0)),
                "answer_relevancy":  float(result.get("answer_relevancy", 0.0)),
                "context_precision": float(result.get("context_precision", 0.0)),
                "reasoning":         result.get("reasoning", ""),
            }
            
            # logger.info(f"RAGAS result: {normalized}")
            return normalized

        except Exception as e:
            result = {
                "faithfulness":      0.0,
                "answer_relevancy":  0.0,
                "context_precision": 0.0,
                "reasoning":         f"Eval skip: {str(e)[:60]}",
            }
            logger.warning(f"RAGAS eval failed: {e}")
            return result 