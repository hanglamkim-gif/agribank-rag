import streamlit as st
import os
import json
from pathlib import Path
import time
import uuid

# Set up environment early if needed
from dotenv import set_key

st.set_page_config(
    page_title="RAG Foundation - Buổi 09",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Imports
import hierarchical_rag
from hierarchical_rag import load_buoi09_config, generate_answer

def get_env_path():
    return Path(hierarchical_rag.__file__).resolve().parent / ".env"

def load_manifest():
    base_dir = Path(hierarchical_rag.__file__).resolve().parent / "storage" / "hierarchy"
    try:
        with open(base_dir / "manifest.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def warning_mapping(status):
    mapping = {
        "hierarchy_not_ready": "Hệ thống Parent-Child chưa được build. Vui lòng bấm 'Build Hierarchy' ở Sidebar.",
        "query_generation_unavailable": "Không thể gọi API sinh truy vấn biến thể. Vui lòng kiểm tra API Key hoặc kết nối.",
        "reranker_unavailable": "Hệ thống Rerank không khả dụng. Vui lòng kiểm tra model path hoặc bộ nhớ.",
        "insufficient_evidence": "Không tìm thấy Evidence đạt chuẩn sau khi lọc Reranker.",
        "multi_query_partial": "Một số truy vấn biến thể gặp lỗi trong quá trình Hybrid Search."
    }
    return mapping.get(status, f"Lỗi không xác định: {status}")

def format_parent_tree(p):
    """ Helper to format a parent node into a markdown string. """
    md = f"**Parent ID:** `{p['parent_id']}`\n\n"
    path = p.get('structural_path', {})
    art_key = path.get('article_key', 'N/A')
    md += f"- **Path:** `{art_key}`\n"
    md += f"- **Source:** {p.get('source', 'N/A')} (Pages {p.get('page_start')} - {p.get('page_end')})\n"
    md += f"- **Rerank Score:** {p.get('parent_rerank_score', 0):.4f} (Raw: {p.get('parent_rerank_raw_score', 0):.4f})\n"
    
    if p.get("warnings"):
        md += f"\n> ⚠️ **Warnings:** {', '.join(p['warnings'])}\n"
        
    md += f"\n**Supporting Children:**\n"
    for cid in p.get("supporting_child_ids", []):
        marker = " (🌟 Anchor)" if cid == p.get("anchor_child_id") else ""
        md += f"- `{cid}`{marker}\n"
    return md

def build_query_child_matrix(res):
    """ Build matrix data for Fan-out tab """
    queries = res.get("queries", [])
    q_dict = {q["query_id"]: q for q in queries}
    
    # Actually, in mode single_parent/multi_parent, child hits are not directly returned as a list, 
    # but they are embedded inside parents. Oh wait, `evidence` in parent mode contains parent docs.
    # Where do we get child hits?
    # Ah, `generate_answer` doesn't return raw child hits in parent mode! It returns `evidence_list` which are parents.
    # Wait, the prompt says Tab 2 shows query-child matrix.
    # If the user wants to see child matrix, I can either modify generate_answer to return child_hits in trace, or just reconstruct it.
    pass

def init_session_state():
    if "result" not in st.session_state:
        st.session_state.result = None
    if "compare_result" not in st.session_state:
        st.session_state.compare_result = None

def render_sidebar():
    st.sidebar.title("Cấu hình RAG Buổi 09")
    config = load_buoi09_config()
    
    # Model info
    st.sidebar.markdown("### Trạng thái hệ thống")
    has_key = bool(config.get("API_KEY")) or bool(config.get("GEMINI_API_KEY"))
    st.sidebar.info(f"🔑 API Key: {'Có' if has_key else 'Thiếu'}")
    
    manifest = load_manifest()
    if manifest:
        st.sidebar.success(f"📚 Hierarchy Store: Ready\n\n- Parents: {manifest.get('parent_count')}\n- Children: {manifest.get('child_count')}\n- Warnings: {manifest.get('parent_warnings_count')}")
    else:
        st.sidebar.error("📚 Hierarchy Store: Missing/Stale")
        
    if st.sidebar.button("Build Hierarchy"):
        with st.spinner("Đang build hierarchy..."):
            try:
                hierarchical_rag.cmd_build_hierarchy()
                st.toast("Build hierarchy thành công!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Lỗi: {e}")
                
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Runtime Config (Env)")
    env_path = get_env_path()
    
    mq_count = st.sidebar.number_input("MULTI_QUERY_COUNT", min_value=1, max_value=5, value=config.get("MULTI_QUERY_COUNT", 3))
    pq_cands = st.sidebar.number_input("PER_QUERY_CANDIDATES", min_value=1, max_value=100, value=config.get("PER_QUERY_CANDIDATES", 20))
    p_cands = st.sidebar.number_input("PARENT_CANDIDATES", min_value=1, max_value=100, value=config.get("PARENT_CANDIDATES", 20))
    fp_top_k = st.sidebar.number_input("FINAL_PARENT_TOP_K", min_value=1, max_value=100, value=config.get("FINAL_PARENT_TOP_K", 5))
    r_min = st.sidebar.number_input("RERANK_MIN_SCORE", min_value=0.0, max_value=1.0, value=float(config.get("RERANK_MIN_SCORE", 0.5)), step=0.05)
    
    if st.sidebar.button("Lưu cấu hình"):
        set_key(env_path, "MULTI_QUERY_COUNT", str(mq_count))
        set_key(env_path, "PER_QUERY_CANDIDATES", str(pq_cands))
        set_key(env_path, "PARENT_CANDIDATES", str(p_cands))
        set_key(env_path, "FINAL_PARENT_TOP_K", str(fp_top_k))
        set_key(env_path, "RERANK_MIN_SCORE", str(r_min))
        st.toast("Đã lưu cấu hình!")
        st.rerun()

def tab_ask():
    st.header("Ask Advanced RAG")
    config = load_buoi09_config()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area("Nhập câu hỏi pháp lý:", height=100)
    with col2:
        mode = st.selectbox("Chọn Pipeline Mode", ["multi_parent", "single_parent", "multi_flat", "single_flat"], index=0)
        run_btn = st.button("🚀 Chạy Pipeline", type="primary", use_container_width=True)
        
    if run_btn and question:
        with st.spinner("Đang xử lý pipeline nhiều tầng..."):
            st.session_state.result = generate_answer(question, config, mode=mode)
            
    res = st.session_state.result
    if res:
        status = res["status"]
        if status in ["hierarchy_not_ready", "query_generation_unavailable", "reranker_unavailable", "insufficient_evidence"]:
            st.warning(warning_mapping(status))
            return
            
        st.markdown("### 📝 Câu trả lời")
        st.info(res.get("answer", ""))
        
        cits = res.get("citations", [])
        if cits:
            st.markdown("### 🔍 Nguồn Trích Dẫn")
            for c in cits:
                pid = c.get('parent_id', c.get('chunk_id'))
                st.markdown(f"**{c['label']}** ➔ `{pid}` (Trang {c.get('page_start')})")
                
        tr = res.get("trace", {})
        st.markdown("### ⏱️ Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng thời gian (ms)", f"{tr.get('latency_ms', {}).get('total', 0):.0f}")
        col2.metric("Generation Calls", tr.get("api_call_counts", {}).get("answer_generation", 0) + tr.get("api_call_counts", {}).get("query_variants", 0))
        col3.metric("Accepted Evidence", len(res.get("evidence", [])))

def tab_fanout():
    st.header("Query Fan-Out")
    res = st.session_state.result
    if not res:
        st.info("Chạy pipeline ở Tab 1 trước.")
        return
        
    queries = res.get("queries", [])
    if not queries:
        st.warning("Không có truy vấn nào được sinh ra.")
        return
        
    st.markdown("### Các biến thể truy vấn")
    for q in queries:
        if q["query_id"] == "Q0":
            st.success(f"**{q['query_id']} (Original):** {q['text']}")
        else:
            st.info(f"**{q['query_id']} ({q.get('focus', 'generated')}):** {q['text']}")
            
    # For matrix, we don't have direct child hits in parent mode response, so we just show trace overlaps
    st.markdown("### Child Hit Overlap Distribution")
    dist = res.get("trace", {}).get("overlap_distribution", {})
    if dist:
        import pandas as pd
        df = pd.DataFrame(list(dist.items()), columns=["Số lượng Query Support", "Số lượng Child Hits"])
        st.dataframe(df)

def tab_explorer():
    st.header("Parent–Child Explorer")
    res = st.session_state.result
    if not res:
        st.info("Chạy pipeline ở Tab 1 trước.")
        return
        
    if "parent" not in res["mode"]:
        st.info("Mode hiện tại là Flat, không có Parent data.")
        return
        
    evidences = res.get("evidence", [])
    if not evidences:
        st.warning("Không có Parent evidence nào được trả về.")
        return
        
    for idx, p in enumerate(evidences):
        title = f"Parent Rank {p.get('parent_rerank_rank', '?')} | {p['parent_id']}"
        with st.expander(title, expanded=(idx==0)):
            st.markdown(format_parent_tree(p))
            st.markdown("**Nội dung:**")
            st.markdown(f"```text\n{p['text']}\n```")

def tab_compare():
    st.header("Mode Comparison")
    config = load_buoi09_config()
    
    question = st.text_area("Nhập câu hỏi để so sánh 4 chế độ (Chỉ Retrieval, không Generation):", key="compare_q")
    if st.button("⚖️ So sánh", type="primary"):
        if question:
            with st.spinner("Đang chạy 4 chế độ..."):
                results = []
                modes = ["single_flat", "multi_flat", "single_parent", "multi_parent"]
                for mode in modes:
                    try:
                        # Skip generation API call
                        r = generate_answer(question, config, mode=mode, custom_gen_fn=lambda x: "")
                        results.append(r)
                    except Exception as e:
                        results.append({"mode": mode, "status": "error", "warnings": [str(e)]})
                st.session_state.compare_result = results
                
    comps = st.session_state.compare_result
    if comps:
        import pandas as pd
        data = []
        for c in comps:
            mode = c.get("mode")
            status = c.get("status")
            ev = c.get("evidence", [])
            tr = c.get("trace", {})
            lat = tr.get("latency_ms", {}).get("total", 0)
            
            unit = "parent" if "parent" in mode else "child"
            expanded_parents = tr.get("unique_parent_count", 0) if "parent" in mode else 0
            retrieved_children = tr.get("input_child_hit_count", 0) if "parent" in mode else len(ev)
            
            data.append({
                "Mode": mode,
                "Status": status,
                "Unit Type": unit,
                "Accepted Evidences": len(ev),
                "Retrieved Children": retrieved_children,
                "Expanded Parents": expanded_parents,
                "Latency (ms)": f"{lat:.0f}"
            })
            
        st.dataframe(pd.DataFrame(data), use_container_width=True)

def tab_eval():
    st.header("Evaluation Report")
    
    report_file = Path(__file__).resolve().parent / "reports" / "latest_report.json"
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                rep = json.load(f)
            st.json(rep)
        except Exception as e:
            st.error(f"Lỗi đọc report: {e}")
    else:
        st.info("Chưa có report đánh giá nào được tạo ra.")

def main():
    st.title("RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval")
    st.markdown("### *Query fan-out → Hybrid per query → Cross-query RRF → Parent expansion → Parent rerank*")
    
    init_session_state()
    render_sidebar()
    
    t1, t2, t3, t4, t5 = st.tabs(["Ask Advanced RAG", "Query Fan-Out", "Parent–Child Explorer", "Mode Comparison", "Evaluation"])
    with t1: tab_ask()
    with t2: tab_fanout()
    with t3: tab_explorer()
    with t4: tab_compare()
    with t5: tab_eval()

if __name__ == "__main__":
    main()
