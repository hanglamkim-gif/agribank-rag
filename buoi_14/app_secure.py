import os
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
from src.secure_retriever import SecureRetriever

ALL_ROLES = ["Admin", "HR_Manager", "Risk_Officer", "Staff", "Guest"]

st.set_page_config(page_title="Secure RAG (RBAC) - Buổi 15", page_icon="🔐", layout="wide")

@st.cache_resource
def load_retriever():
    return SecureRetriever(BASE_DIR / "data" / "processed" / "chunks_secure.csv")

retriever = load_retriever()

st.title("🔐 Secure RAG Search với RBAC — Buổi 15")

with st.sidebar:
    st.header("👤 Kiểm soát Vai trò (RBAC)")
    selected_roles = st.multiselect(
        "Chọn vai trò của bạn:",
        options=ALL_ROLES,
        default=["Staff"]
    )
    st.divider()
    st.header("⚙️ Cấu hình Retrieval")
    method = st.selectbox("Phương pháp:", ("Hybrid + Rerank", "Hybrid (RRF)", "BM25 Only", "Dense Only"))
    top_k = st.slider("Top-k kết quả:", 1, 3, 2)

query = st.text_input("Nhập câu hỏi tra cứu:", value="Nghị định 123 sửa đổi bổ sung")

if st.button("🚀 Tìm kiếm an toàn", type="primary"):
    if not selected_roles:
        st.error("⚠️ Vui lòng chọn ít nhất 1 vai trò ở thanh bên trái!")
    elif not query.strip():
        st.warning("Vui lòng nhập nội dung câu hỏi.")
    else:
        results, blocked_count = retriever.search(query, selected_roles, method=method, top_k=top_k)
        
        if blocked_count > 0:
            st.info(f"🛡️ **Bộ lọc RBAC:** Đã loại bỏ {blocked_count} tài liệu ngoài phạm vi quyền xem của bạn.")

        if not results:
            st.warning("Không tìm thấy tài liệu phù hợp trong phạm vi quyền truy cập của bạn.")
        else:
            st.subheader(f"💡 Kết quả truy xuất ({method})")
            for item in results:
                rank_val = item.get("final_rank", item.get("rank"))
                with st.expander(f"Top {rank_val}: {item['citation']} — 🔑 Quyền xem: {item.get('allowed_roles')}"):
                    st.write(item["text"])
                    st.caption(f"**Doc ID:** {item['document_id']} | **Method:** {item['retrieval_method']}")
                    st.markdown("**🕸️ Graph Hints (Neo4j):**")
                    hints = retriever.get_secure_graph_hints(item["document_id"], selected_roles)
                    for h in hints:
                        st.code(h, language="text")