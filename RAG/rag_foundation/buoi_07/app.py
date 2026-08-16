"""
Web UI cho Hệ thống Tra cứu Nghiệp vụ Ngân hàng (RAG Buổi 07).
"""

import sys
import os
import asyncio
from pathlib import Path

# Khắc phục lỗi ConnectionResetError / WinError 10054 trên Python Windows
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import rag

# ----------------- CẤU HÌNH TRANG STREAMLIT -----------------
st.set_page_config(
    page_title="Trợ lý Nhân sự (Advanced RAG)",
    page_icon="🏦",
    layout="wide"
)

cfg = rag.load_config()

# ----------------- SIDEBAR CẤU HÌNH -----------------
with st.sidebar:
    st.header("⚙️ Cấu hình RAG")
    
    strategy = st.selectbox(
        "Chọn Strategy Chunks:",
        options=["hierarchical", "semantic", "fixed-size"],
        index=0,
        key="select_strategy_sidebar",
        help="Lựa chọn chiến lược phân đoạn văn bản đã thực hiện ở Buổi 05."
    )
    
    top_k = st.slider(
        "Top-K (Số chunk truy xuất):",
        min_value=1,
        max_value=15,
        value=cfg.get("default_top_k", 5),
        key="slider_topk_sidebar",
        help="Số lượng đoạn văn bản liên quan nhất được truy xuất."
    )

    rag_max_distance = st.slider(
        "Max Distance (Ngưỡng tin cậy):",
        min_value=0.1,
        max_value=1.0,
        value=cfg.get("rag_max_distance", 0.35),
        step=0.05,
        key="slider_maxdist_sidebar",
        help="Ngưỡng khoảng cách tối đa. Càng nhỏ càng khắt khe."
    )
    
    # Cập nhật config động theo slider
    cfg["rag_max_distance"] = rag_max_distance
    cfg["default_top_k"] = top_k

    st.markdown("---")
    st.subheader("📊 Trạng thái Hệ thống")

    status_info = rag.run_status(strategy=strategy, _config=cfg)
    record_count = status_info.get("record_count", 0)

    st.write(f"**Collection:** `{status_info.get('collection_name')}`")
    st.write(f"**Embedding:** `{status_info.get('embedding_model')}` ({status_info.get('embedding_dim')}d)")
    st.write(f"**Generation:** `{status_info.get('generation_model')}`")
    st.write(f"**Số bản ghi:** `{record_count}`")
    st.write(f"**Ngưỡng Max Distance:** `{rag_max_distance}`")

    if cfg.get("api_key"):
        st.success("🔑 API Key: Đã sẵn sàng")
    else:
        st.error("❌ API Key: Chưa cấu hình GEMINI_API_KEY")

    st.markdown("---")
    btn_index = st.button(
        "🔄 Index Dữ liệu Chunks",
        key="btn_trigger_index_sidebar",
        use_container_width=True
    )
    if btn_index:
        with st.spinner(f"Đang index strategy '{strategy}'..."):
            try:
                res_idx = rag.run_index(strategy=strategy, reset=True, _config=cfg)
                st.success(f"Đã index thành công {res_idx.get('indexed_chunks', 0)} chunks!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi index: {e}")

# ----------------- GIAO DIỆN CHÍNH -----------------
st.title("👨‍💼 Trợ lý Tra cứu Quy định Nhân sự (Advanced RAG)")
st.caption("Hệ thống RAG bảo đảm Grounding dữ liệu, kiểm soát Confidence Gate & Mapping Citation chính xác.")

question = st.text_area(
    "Nhập câu hỏi liên quan đến chính sách nhân sự:",
    value="Lao động nữ sinh con được nghỉ thai sản bao nhiêu tháng?",
    height=110,
    key="input_query_text"
)

col1, col2 = st.columns([1, 5])
with col1:
    btn_submit = st.button(
        "🔍 Gửi câu hỏi",
        key="btn_submit_query_main",
        type="primary",
        use_container_width=True
    )

if btn_submit:
    if not question.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi nghiệp vụ.")
    elif record_count == 0:
        st.warning(
            f"⚠️ Collection `{status_info.get('collection_name')}` hiện chưa có dữ liệu (Số bản ghi = 0). "
            "Vui lòng nhấn nút **'🔄 Index Dữ liệu Chunks'** ở thanh menu bên trái trước!"
        )
    else:
        with st.spinner("Đang truy xuất dữ liệu và tổng hợp câu trả lời..."):
            try:
                result = rag.run_query(
                    question=question.strip(),
                    top_k=top_k,
                    strategy=strategy,
                    _config=cfg
                )

                st.markdown("---")
                st.markdown("### 📝 Câu trả lời:")
                st.markdown(result.get("answer", ""))

                citations = result.get("citations", [])
                if citations:
                    st.markdown("### 📚 Trích dẫn nguồn:")
                    for cit in citations:
                        st.info(f"📌 {cit.get('display', '')}")

                with st.expander("🔍 Chi tiết các đoạn Evidence truy xuất (Confidence Gate)"):
                    evidences = result.get("evidence", [])
                    for ev in evidences:
                        status_badge = "✅ [ĐẠT NGƯỠNG]" if ev.get("accepted") else "❌ [VƯỢT NGƯỠNG]"
                        st.markdown(
                            f"**{status_badge} {ev.get('evidence_id')}** | Khoảng cách: `{ev.get('distance')}` | "
                            f"Nguồn: `{ev.get('source')}` (tr. {ev.get('page_start')}-{ev.get('page_end')})"
                        )
                        st.code(ev.get("text", ""), language="text")

                if result.get("warnings"):
                    st.warning("⚠️ Cảnh báo: " + " | ".join(result["warnings"]))

            except Exception as err:
                st.error(f"❌ Lỗi khi thực thi: {err}")