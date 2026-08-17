import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, corpus_df: pd.DataFrame, cache_dir: Path, model_name: str = "keepitreal/vietnamese-sbert"):
        self.df = corpus_df.reset_index(drop=True)
        self.cache_file = cache_dir / "dense_embeddings.pkl"
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_create_embeddings(cache_dir)

    def _load_or_create_embeddings(self, cache_dir: Path):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        
        embeddings = self.model.encode(self.df["text"].tolist(), show_progress_bar=True, normalize_embeddings=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

    def search(self, query: str, top_k: int = 5):
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "text": row["text"],
                "score": float(scores[idx]),
                "retrieval_method": "Dense",
                "citation": row["citation"]
            })
        return results