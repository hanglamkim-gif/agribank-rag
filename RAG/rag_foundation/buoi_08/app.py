import streamlit as st
import os
import json
import time
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from advanced_rag import load_advanced_config, advanced_generate_answer, show_status
from rag import get_collection_name, get_chroma_client, STORAGE_DIR

# ---------------------------------------------------------
# STATE & CACHING
# ---------------------------------------------------------
import rag
import advanced_rag

# Cache original load_chunks
_original_load_chunks = rag.load_chunks

@st.cache_data(show_spinner=False)
def cached_load_chunks(strategy, *args, **kwargs):
    return _original_load_chunks(strategy, *args, **kwargs)

# Monkey-patch to use cached version globally
rag.load_chunks = cached_load_chunks

@st.cache_resource(show_spinner=False)
def preload_reranker(config):
    # This just ensures we only load it once and keep it in st.cache_resource as well
    # Though advanced_rag also has its own globals.
    return advanced_rag.load_reranker_model(config)

def init_session():
    if "query_result" not in st.session_state:
        st.session_state.query_result = None
    if "compare_result" not in st.session_state:
        st.session_state.compare_result = None

init_session()
config = load_advanced_config()

# ---------------------------------------------------------
# UI CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced RAG", layout="wide")
st.title("Advanced RAG - Hệ thống Truy vấn Đa tầng")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("Cấu hình & Trạng thái")
    strategy = st.selectbox("Strategy", ["hierarchical", "semantic", "fixed-size"], index=0)
    
    st.subheader("Retrieval Params")
    st.text(f"Final Top-K: {config.get('FINAL_TOP_K', 5)}")
    st.text(f"BM25 Candidates: {config.get('BM25_CANDIDATES', 20)}")
    st.text(f"Semantic Candidates: {config.get('SEMANTIC_CANDIDATES', 20)}")
    
    st.subheader("Fusion (RRF)")
    st.text(f"RRF k: {config.get('RRF_K', 60)}")
    st.text(f"Weights: BM25={config.get('RRF_BM25_WEIGHT', 1.0)}, Sem={config.get('RRF_SEMANTIC_WEIGHT', 1.0)}")
    
    st.subheader("Reranker")
    model_name = config.get('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
    st.text(f"Model: {model_name}")
    st.text(f"Device: {config.get('RERANK_DEVICE', 'auto')}")
    st.text(f"Candidates K: {config.get('RERANK_CANDIDATES', 20)}")
    st.text(f"Min Score: {config.get('RERANK_MIN_SCORE', 0.50)}")
    
    # Check cache status
    cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
    cache_exists = os.path.exists(cache_dir)
    st.text(f"Cache Exists: {'✅' if cache_exists else '❌'}")
    
    st.subheader("Storage")
    c_name = get_collection_name(strategy, config.get('GEMINI_EMBEDDING_DIM', 768), config.get('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-2'))
    client = get_chroma_client(STORAGE_DIR)
    col_count = 0
    if client:
        try:
            col_count = client.get_collection(c_name).count()
        except:
            pass
    st.text(f"Collection: {c_name[:15]}...")
    st.text(f"Count: {col_count}")
    
    api_key = config.get("GEMINI_API_KEY", "")
    st.text(f"API Key: {'✅ Có' if api_key else '❌ Thiếu'}")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Hỏi đáp Advanced RAG", "So sánh Retrieval", "Pipeline Trace", "Đánh giá"])

with tab1:
    st.header("Hỏi đáp Advanced RAG")
    question = st.text_input("Nhập câu hỏi pháp lý:", value="Điều 7 quy định gì?")
    col_mode, col_btn = st.columns([2, 1])
    with col_mode:
        mode = st.selectbox("Retrieval Mode", ["hybrid_rerank", "hybrid", "semantic", "bm25"], index=0)
    with col_btn:
        st.write("") # padding
        st.write("")
        if st.button("Truy vấn", type="primary"):
            with st.spinner(f"Đang xử lý pipeline ({mode})..."):
                try:
                    res = advanced_generate_answer(question, strategy, mode=mode)
                    st.session_state.query_result = res
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}")

    # Display results
    if st.session_state.query_result:
        res = st.session_state.query_result
        if res["question"] == question: # Only show if matches current input
            st.subheader(f"Trạng thái: {res['status']}")
            
            if res["status"] == "reranker_unavailable":
                st.error("❌ Mô hình reranker chưa được tải hoặc không khả dụng.")
                st.info("Vui lòng kích hoạt tải model hoặc chạy `python advanced_rag.py rerank ...` trong terminal để tải model lần đầu.")
            elif res["status"] == "insufficient_evidence":
                st.warning("⚠️ Không có bằng chứng nào thỏa mãn ngưỡng tin cậy.")
            
            if res["warnings"]:
                for w in res["warnings"]:
                    st.warning(f"Cảnh báo: {w}")
                    
            if res["answer"]:
                st.markdown("### Câu trả lời")
                st.info(res["answer"])
                
            if res["citations"]:
                st.markdown("### Trích dẫn")
                for c in res["citations"]:
                    st.markdown(f"**{c['label']}** Chunk `{c['chunk_id']}` - {c['source']} (Trang {c['page_start']})")
                    
            st.markdown("### Bằng chứng (Evidences)")
            for i, ev in enumerate(res["evidence"]):
                accepted = "✅" if ev["accepted"] else "❌"
                with st.expander(f"[{'E'+str(i+1) if ev['accepted'] else 'REJECTED'}] Chunk {ev['chunk_id']} {accepted}"):
                    st.markdown(f"**Source:** {ev['source']} (Page {ev['page_start']})")
                    st.markdown(f"**BM25 Rank/Score:** {ev['bm25_rank']} / {ev['bm25_score']}")
                    st.markdown(f"**Semantic Rank/Dist:** {ev['semantic_rank']} / {ev['semantic_distance']}")
                    st.markdown(f"**Fused Rank/Score:** {ev['fused_rank']} / {ev['rrf_score']}")
                    st.markdown(f"**Rerank Rank/Score/Change:** {ev['rerank_rank']} / {ev['rerank_score']} / {ev.get('rank_change')}")
                    st.markdown("---")
                    st.text(ev["text"])

with tab2:
    st.header("So sánh Retrieval")
    st.markdown("Chạy cùng một câu hỏi qua 4 chế độ để xem sự thay đổi thứ hạng. Quá trình này **không** gọi Generation LLM.")
    
    comp_question = st.text_input("Câu hỏi so sánh:", value="Điều 7 quy định gì?", key="comp_q")
    if st.button("So sánh Rank"):
        with st.spinner("Đang chạy 4 chế độ retrieval..."):
            modes = ["bm25", "semantic", "hybrid", "hybrid_rerank"]
            results_by_mode = {}
            for m in modes:
                try:
                    res = advanced_generate_answer(comp_question, strategy, mode=m, custom_gen_fn=lambda x: "skipped")
                    results_by_mode[m] = res["evidence"]
                except Exception as e:
                    st.error(f"Lỗi ở chế độ {m}: {e}")
                    results_by_mode[m] = []
            
            all_chunks = {}
            for m in modes:
                for idx, ev in enumerate(results_by_mode[m], start=1):
                    cid = ev["chunk_id"]
                    if cid not in all_chunks:
                        all_chunks[cid] = {"text": ev["text"], "ranks": {}}
                    all_chunks[cid]["ranks"][m] = idx
            
            table_data = []
            for cid, data in all_chunks.items():
                r = data["ranks"]
                table_data.append({
                    "chunk_id": cid,
                    "bm25_rank": r.get("bm25"),
                    "semantic_rank": r.get("semantic"),
                    "fused_rank": r.get("hybrid"),
                    "rerank_rank": r.get("hybrid_rerank"),
                    "rank_change": r.get("hybrid") - r.get("hybrid_rerank") if r.get("hybrid") and r.get("hybrid_rerank") else None,
                    "preview": data["text"][:100] + "..."
                })
            
            st.session_state.compare_result = pd.DataFrame(table_data)

    if st.session_state.compare_result is not None:
        df = st.session_state.compare_result
        st.dataframe(
            df,
            column_config={
                "chunk_id": "Chunk ID",
                "bm25_rank": st.column_config.NumberColumn("BM25", format="%d"),
                "semantic_rank": st.column_config.NumberColumn("Semantic", format="%d"),
                "fused_rank": st.column_config.NumberColumn("Hybrid Fused", format="%d"),
                "rerank_rank": st.column_config.NumberColumn("Hybrid Rerank", format="%d"),
                "rank_change": st.column_config.NumberColumn("Change", format="%+d"),
                "preview": "Nội dung"
            },
            hide_index=True,
            use_container_width=True
        )

with tab3:
    st.header("Pipeline Trace")
    if st.session_state.query_result:
        t = st.session_state.query_result["trace"]
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("BM25 Candidates", t.get("bm25_candidates", 0))
        c2.metric("Semantic Candidates", t.get("semantic_candidates", 0))
        c3.metric("Union / Overlap", f"{t.get('union',0)} / {t.get('overlap',0)}")
        c4.metric("Reranked", t.get("reranked", 0))
        c5.metric("Accepted", t.get("accepted", 0))
        
        st.markdown("### Thời gian xử lý (Latency)")
        lat = t.get("latency_ms", {})
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        l1.metric("BM25", f"{lat.get('bm25', 0):.1f} ms")
        l2.metric("Semantic", f"{lat.get('semantic', 0):.1f} ms")
        l3.metric("Fusion", f"{lat.get('fusion', 0):.1f} ms")
        l4.metric("Rerank", f"{lat.get('rerank', 0):.1f} ms")
        l5.metric("Generation", f"{lat.get('generation', 0):.1f} ms")
        l6.metric("Total", f"{lat.get('total', 0):.1f} ms")
        
        st.info("""
        **Chú thích:**
        - BM25 score cao hơn tốt hơn.
        - Cosine distance thấp hơn tốt hơn.
        - RRF/rerank score cao hơn tốt hơn.
        - Rerank score (Sigmoid) không phải là xác suất đúng tuyệt đối.
        """)
    else:
        st.write("Vui lòng thực hiện truy vấn ở tab 'Hỏi đáp Advanced RAG' trước.")

with tab4:
    st.header("Đánh giá Hệ thống")
    eval_dir = Path(STORAGE_DIR).parent.parent / "reports"
    report_files = sorted(list(eval_dir.glob("*.json")))
    
    if not report_files:
        st.info("Chưa có báo cáo đánh giá. Vui lòng chạy `evaluate.py` trước.")
    else:
        selected_report = st.selectbox("Chọn báo cáo:", [f.name for f in report_files])
        report_path = eval_dir / selected_report
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            
            st.subheader("Bảng metrics")
            if "metrics" in report_data:
                st.json(report_data["metrics"])
            else:
                st.write(report_data)
                
        except Exception as e:
            st.error(f"Không thể đọc báo cáo: {e}")
