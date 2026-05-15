from backend.rag.components import (CleanerDispatcher,ParserDispatcher,
                                    get_language_or_fallback,is_supported_language,detect_language)
from backend.rag.pipelines.ingestion_pipeline import IngestionPipeline
from backend.rag.pipelines.retrieval_pipeline import RetrievalPipeline
