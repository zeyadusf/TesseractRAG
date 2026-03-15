def loop_score(results: list[dict], k: int, accu_score: dict, chunk_lookup: dict):
    for idx, chunk in enumerate(results):
        rank = idx + 1
        rrf_score = 1 / (k + rank)
        chunk_id = chunk['content']  # string key
        if chunk_id in accu_score:
            accu_score[chunk_id] += rrf_score
        else:
            accu_score[chunk_id] = rrf_score
            chunk_lookup[chunk_id] = chunk  # store original dict

def reciprocal_rank_fusion( 
    bm25_results: list[dict],
    faiss_results: list[dict],
    k: int = 60) -> list[dict]:
    
    accu_score = dict()
    chunk_lookup = dict()
    loop_score(bm25_results, k, accu_score,chunk_lookup)
    loop_score(faiss_results, k, accu_score,chunk_lookup)

    accu_score = sorted(
    accu_score.items(),
    key=lambda x: x[1],
    reverse=True
)
    return [chunk_lookup[chunk_id] for chunk_id, score in accu_score]

class HybridRetriever:
    def __init__(self, bm25_retriever, faiss_indexer, embedder, chunks: list[dict]):
        self.bm25_retriever = bm25_retriever
        self.faiss_indexer = faiss_indexer
        self.embedder = embedder
        self.chunks = chunks

    def retrieve(self, query: str, strategy: str = "hybrid", top_k: int = 10) -> list[dict]:
        
        bm25_res = self.bm25_retriever.retrieve(query,top_k)
        query_vec = self.embedder.embed_query(query)
        scores, indices = self.faiss_indexer.search(query_vec, 5)
        faiss_res = [self.chunks[i] for i in indices[0]]
        if strategy == 'hybrid' :
            return reciprocal_rank_fusion(bm25_res,faiss_res)
        elif strategy == 'semantic' :
            return faiss_res
        elif strategy == 'lexical' :
            return bm25_res


if __name__ == "__main__":
    # --- TEST ---
    bm25 = [
        {'content': 'Render and Railway are supported platforms', 'document_name': 'a.pdf'},
        {'content': 'HuggingFace Spaces hosts the frontend', 'document_name': 'a.pdf'},
        {'content': 'Docker is used for containerization', 'document_name': 'b.pdf'},
    ]

    faiss = [
        {'content': 'HuggingFace Spaces hosts the frontend', 'document_name': 'a.pdf'},
        {'content': 'Render and Railway are supported platforms', 'document_name': 'a.pdf'},
        {'content': 'FAISS enables fast vector search', 'document_name': 'b.pdf'},
    ]

    results = reciprocal_rank_fusion(bm25, faiss)

    for i, chunk in enumerate(results):
        print(f"Rank {i+1}: {chunk['content']}")
