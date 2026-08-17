import json
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / ".env")

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import NeuralReranker

class SecureRetriever:
    def __init__(self, secure_corpus_path: Path):
        self.raw_df = pd.read_csv(secure_corpus_path)
        self.raw_df["allowed_roles"] = self.raw_df["allowed_roles_str"].apply(json.loads)
        self.reranker = NeuralReranker()

    def filter_by_role(self, user_roles: list) -> pd.DataFrame:
        user_roles_set = set(user_roles)
        filtered_df = self.raw_df[
            self.raw_df["allowed_roles"].apply(lambda doc_roles: bool(set(doc_roles).intersection(user_roles_set)))
        ].copy()
        return filtered_df.reset_index(drop=True)

    def search(self, query: str, user_roles: list, method: str = "Hybrid + Rerank", top_k: int = 3):
        filtered_df = self.filter_by_role(user_roles)
        if filtered_df.empty:
            return [], len(self.raw_df)

        total_blocked = len(self.raw_df) - len(filtered_df)
        effective_k = min(top_k, len(filtered_df))

        bm25 = BM25Retriever(filtered_df)
        dense = DenseRetriever(filtered_df)
        hybrid = HybridRetriever(bm25, dense)

        if method == "BM25 Only":
            results = bm25.search(query, top_k=effective_k)
        elif method == "Dense Only":
            results = dense.search(query, top_k=effective_k)
        elif method == "Hybrid (RRF)":
            results = hybrid.search(query, candidate_k=len(filtered_df), top_k=effective_k)
        else:
            candidates = hybrid.search(query, candidate_k=len(filtered_df), top_k=len(filtered_df))
            results = self.reranker.rerank(query, candidates, top_k=effective_k)

        for r in results:
            match_row = filtered_df[filtered_df["chunk_id"].astype(str) == str(r["chunk_id"])]
            if not match_row.empty:
                r["allowed_roles"] = match_row.iloc[0]["allowed_roles"]

        return results, total_blocked

    def get_secure_graph_hints(self, doc_id: str, user_roles: list):
        try:
            driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))
            )
            with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
                res = session.run("""
                    MATCH (v:VanBan {id: $doc_id, lab_session: 'buoi_14'})-[r]->(target)
                    WHERE any(role IN v.allowed_roles WHERE role IN $user_roles)
                    RETURN type(r) AS rel, labels(target)[0] AS target_label, target.id AS target_id
                    LIMIT 5
                """, doc_id=doc_id, user_roles=user_roles)
                records = [f"({doc_id}) -[:{r['rel']}]-> ({r['target_label']}: {r['target_id']})" for r in res]
            driver.close()
            return records if records else ["Không có liên kết đồ thị hoặc bị ẩn do giới hạn quyền."]
        except Exception:
            return ["Neo4j chưa sẵn sàng kết nối."]