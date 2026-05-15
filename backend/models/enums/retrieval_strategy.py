from enum import Enum

class RetrievalStrategy(str, Enum):
    """Retrieval strategy options for the RAG pipeline."""
    AUTO = "auto"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    LEXICAL = "lexical"