from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, corpus_df: pd.DataFrame, cache_dir: Path = None, model_name: str = "keepitreal/vietnamese-sbert"):
        self.df = corpus_df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        if not self.df.empty:
            self.embeddings = self.model.encode(
                self.df["text"].tolist(),
                show_progress_bar=False,
                normalize_embeddings=True
            )
        else:
            self.embeddings = np.array([])

    def search(self, query: str, top_k: int = 5):
        if self.df.empty or len(self.embeddings) == 0:
            return []

        effective_k = min(top_k, len(self.df))
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:effective_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "text": str(row["text"]),
                "score": float(scores[idx]),
                "retrieval_method": "Dense",
                "citation": str(row.get("citation", ""))
            })
        return results