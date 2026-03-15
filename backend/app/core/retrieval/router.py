import re
from app.utils.logger import get_logger

logger = get_logger(__name__)

CONCEPTUAL_STARTERS = ["what is", "what are", "explain", "describe", "how does", "why is"]

class RetrievalRouter:
    def route(self, query: str, user_strategy: str = "auto") -> str:
        logger.info("Router selection strategy now..")
        query = query.strip().lower()
        match = re.search(r'\b([A-Z]{2,}|v\d+|\d{3,})\b', query, re.IGNORECASE)
        if user_strategy != 'auto':
            return user_strategy
        elif len(query.split()) <= 3 and match:
                return 'lexical'
        elif len(query.split()) > 5 and any(query.startswith(s) for s in CONCEPTUAL_STARTERS):
            return 'semantic'
        else : return  'hybrid'
