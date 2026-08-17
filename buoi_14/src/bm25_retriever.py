import re
import pandas as pd
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus_df: pd.DataFrame):
        self.df = corpus_df.reset_index(drop=True)
        self.tokenized_corpus = [self._tokenize(t) for t in self.df["text"]]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str):
        return re.findall(r"\b[\w\-]+\b", str(text).lower())

    def search(self, query: str, top_k: int = 5):
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "text": row["text"],
                "score": float(scores[idx]),
                "retrieval_method": "BM25",
                "citation": row["citation"]
            })
        return results