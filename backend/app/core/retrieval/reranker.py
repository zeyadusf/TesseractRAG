import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_reranker_model = None
_reranker_tokenizer = None


def get_reranker():
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None :
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(get_settings().RERANKER_MODEL)
        _reranker_tokenizer = AutoTokenizer.from_pretrained(get_settings().RERANKER_MODEL)
        logger.info(f"Loading reranker model: {get_settings().RERANKER_MODEL}")
    return _reranker_model , _reranker_tokenizer

class CrossEncoderReranker:
    def __init__(self):
        self.reranker_model , self.reranker_tokenizer = get_reranker()

    def rerank(self, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        pairs = [[query,chunk['content']] for chunk in chunks]
        tokenized_pairs = self.reranker_tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs = self.reranker_model(**tokenized_pairs)
        scores = outputs.logits.squeeze(-1).tolist()
        scored = zip(chunks, scores)          # pairs each chunk with its score
        sorted_chunks = sorted(scored, key=lambda x: x[1], reverse=True)  # sort by score
        return [chunk for chunk, score in sorted_chunks[:top_k]]           # return just the dicts


