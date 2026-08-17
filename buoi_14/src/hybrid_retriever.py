class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, candidate_k: int = 20, top_k: int = 5):
        bm25_res = self.bm25.search(query, top_k=candidate_k)
        dense_res = self.dense.search(query, top_k=candidate_k)

        scores = {}
        info_map = {}

        for item in bm25_res:
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + item["rank"]))
            info_map[cid] = {
                "chunk_id": cid,
                "document_id": item["document_id"],
                "text": item["text"],
                "citation": item["citation"],
                "bm25_rank": item["rank"],
                "dense_rank": None
            }

        for item in dense_res:
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + item["rank"]))
            if cid in info_map:
                info_map[cid]["dense_rank"] = item["rank"]
            else:
                info_map[cid] = {
                    "chunk_id": cid,
                    "document_id": item["document_id"],
                    "text": item["text"],
                    "citation": item["citation"],
                    "bm25_rank": None,
                    "dense_rank": item["rank"]
                }

        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        hybrid_results = []
        for rank, (cid, rrf_score) in enumerate(sorted_chunks, 1):
            entry = info_map[cid]
            hybrid_results.append({
                "final_rank": rank,
                "chunk_id": cid,
                "document_id": entry["document_id"],
                "bm25_rank": entry["bm25_rank"],
                "dense_rank": entry["dense_rank"],
                "rrf_score": round(rrf_score, 5),
                "text": entry["text"],
                "citation": entry["citation"],
                "retrieval_method": "Hybrid_RRF"
            })
        return hybrid_results