import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import NeuralReranker

st.set_page_config(page_title="RAG Hybrid Search - Buổi 14", page_icon="🔍", layout="wide")

@st.cache_resource
def load_retrievers():
    df = pd.read_csv(BASE_DIR / "data" / "processed" / "chunks_normalized.csv")
    bm25 = BM25Retriever(df)
    dense = DenseRetriever(df, cache_dir=BASE_DIR / "cache")
    hybrid = HybridRetriever(bm25, dense)
    reranker = NeuralReranker()
    return bm25, dense, hybrid, reranker

bm25, dense, hybrid, reranker = load_retrievers()

def get_graph_hints(doc_id):
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            res = session.run("""
                MATCH (v:VanBan {id: $doc_id, lab_session: 'buoi_14'})-[r]->(target)
                RETURN type(r) AS rel, labels(target)[0] AS target_label, target.id AS target_id
                LIMIT 5
            """, doc_id=doc_id)
            records = [f"({doc_id}) -[:{r['rel']}]-> ({r['target_label']}: {r['target_id']})" for r in res]
        driver.close()
        return records if records else ["Không có liên kết đồ thị trực tiếp."]
    except Exception:
        return ["Neo4j chưa sẵn sàng kết nối."]

st.title("🔍 RAG Hybrid Search & Mini Knowledge Graph — Buổi 14")

with st.sidebar:
    st.header("⚙️ Tùy chọn Retrieval")
    method = st.selectbox("Phương pháp:", ("Hybrid + Rerank", "Hybrid (RRF)", "BM25 Only", "Dense Only"))
    top_k = st.slider("Số lượng Top-k kết quả:", 1, 3, 2)

query = st.text_input("Nhập câu hỏi quy định / số hiệu điều khoản:", value="Nghị định 123 sửa đổi bổ sung những gì?")

if st.button("🚀 Tìm kiếm", type="primary"):
    if not query.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        st.divider()
        if method == "BM25 Only":
            results = bm25.search(query, top_k=top_k)
        elif method == "Dense Only":
            results = dense.search(query, top_k=top_k)
        elif method == "Hybrid (RRF)":
            results = hybrid.search(query, candidate_k=10, top_k=top_k)
        else:
            candidates = hybrid.search(query, candidate_k=10, top_k=5)
            results = reranker.rerank(query, candidates, top_k=top_k)
            
            st.subheader("📊 Bảng so sánh Xếp hạng Trước & Sau Reranking")
            comp = [{"Chunk ID": x["chunk_id"], "Vị trí Hybrid ban đầu": x["hybrid_rank"], "Vị trí sau Rerank": x["final_rank"], "Điểm Rerank": round(x["rerank_score"], 4)} for x in results]
            st.table(pd.DataFrame(comp))

        st.subheader(f"💡 Kết quả truy xuất ({method})")
        for item in results:
            rank_val = item.get("final_rank", item.get("rank"))
            with st.expander(f"Top {rank_val}: {item['citation']}"):
                st.write(item["text"])
                st.caption(f"**Document ID:** {item['document_id']} | **Method:** {item['retrieval_method']}")
                st.markdown("**🕸️ Graph Hints (Neo4j):**")
                hints = get_graph_hints(item["document_id"])
                for h in hints:
                    st.code(h, language="text")