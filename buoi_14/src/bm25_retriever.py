import re
import pandas as pd
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, corpus_df: pd.DataFrame):
        self.df = corpus_df.reset_index(drop=True)
        if not self.df.empty:
            self.tokenized_corpus = [self._tokenize(t) for t in self.df["text"]]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def _tokenize(self, text: str):
        return re.findall(r"\b[\w\-]+\b", str(text).lower())

    def search(self, query: str, top_k: int = 5):
        if self.df.empty or self.bm25 is None:
            return []

        effective_k = min(top_k, len(self.df))
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:effective_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "text": str(row["text"]),
                "score": float(scores[idx]),
                "retrieval_method": "BM25",
                "citation": str(row.get("citation", ""))
            })
        return results