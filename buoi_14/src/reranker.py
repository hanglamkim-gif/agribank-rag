from sentence_transformers import CrossEncoder

class NeuralReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
            self.is_fallback = False
            self.model_name = model_name
        except Exception:
            self.is_fallback = True
            self.model_name = "FALLBACK_OVERLAP"

    def rerank(self, query: str, candidates: list, top_k: int = 5):
        if not candidates:
            return []

        if not self.is_fallback:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.model.predict(pairs)
        else:
            q_words = set(query.lower().split())
            scores = [len(q_words.intersection(set(c["text"].lower().split()))) for c in candidates]

        scored = list(zip(candidates, scores))
        sorted_candidates = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

        reranked = []
        for rank, (item, score) in enumerate(sorted_candidates, 1):
            reranked.append({
                "final_rank": rank,
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "hybrid_rank": item.get("final_rank"),
                "hybrid_score": item.get("rrf_score"),
                "rerank_score": float(score),
                "text": item["text"],
                "citation": item["citation"],
                "retrieval_method": "Hybrid+Rerank"
            })
        return reranked