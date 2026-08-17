import streamlit as st
import sys
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để import module
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from scripts.graph_rag import answer_with_graph_rag

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Agribank AI Assistant - Graph RAG",
    page_icon="🏦",
    layout="wide"
)

# Header ứng dụng
st.title("🏦 Agribank Enterprise AI Assistant")
st.caption("Hệ thống Trợ lý Nghiệp vụ & Quản trị Rủi ro Đa tầng - Buổi 13 (Graph RAG)")

# Thanh điều khiển bên trái (Sidebar)
with st.sidebar:
    st.header("⚙️ Chế độ Xử lý Tri thức")
    rag_mode = st.radio(
        "Lựa chọn cơ chế RAG:",
        ("🕸️ Graph RAG (Đồ thị Neo4j)", "📄 Vector RAG (Văn bản ChromaDB)"),
        index=0
    )
    st.divider()
    if "Graph RAG" in rag_mode:
        st.success("✅ Đang kích hoạt: **Đồ thị tri thức (Knowledge Graph)**")
        st.markdown("""
        * **Nguồn dữ liệu:** Cơ sở dữ liệu đồ thị Neo4j.
        * **Liên kết:** `Rủi ro` ➔ `Biện pháp kiểm soát` ➔ `Sự kiện biểu hiện`.
        * **Mô hình AI:** Google Gemini.
        """)
    else:
        st.info("ℹ️ Đang kích hoạt: **Tìm kiếm ngữ nghĩa (ChromaDB)**")
        st.markdown("* Dựa trên tập dữ liệu quy trình & văn bản nội bộ dạng văn bản.")

# Khung nhập câu hỏi nghiệp vụ
st.subheader("❓ Đặt câu hỏi nghiệp vụ hoặc tình huống rủi ro")
sample_query = "Hãy phân tích rủi ro hạch toán sai giao dịch chuyển tiền và các biện pháp kiểm soát tương ứng."

user_query = st.text_area(
    "Nội dung câu hỏi:",
    value=sample_query,
    height=90,
    placeholder="Nhập câu hỏi liên quan đến quy trình, rủi ro hoặc kiểm soát..."
)

# Nút thực thi
col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    submit_btn = st.button("🚀 Phân tích ngay", type="primary", use_container_width=True)

if submit_btn:
    if not user_query.strip():
        st.warning("⚠️ Vui lòng nhập câu hỏi trước khi phân tích.")
    else:
        if "Graph RAG" in rag_mode:
            with st.spinner("Đang truy xuất đồ thị Neo4j và tổng hợp phân tích..."):
                try:
                    answer, graph_context = answer_with_graph_rag(user_query)
                    
                    # Chia 2 cột hiển thị kết quả
                    col_result, col_graph = st.columns([3, 2])
                    
                    with col_result:
                        st.subheader("💡 Báo cáo Phân tích từ Trợ lý AI")
                        st.markdown(answer)
                    
                    with col_graph:
                        st.subheader("🕸️ Ngữ cảnh Đồ thị Tri thức (Neo4j)")
                        st.info(graph_context)
                        with st.expander("📌 Xem chi tiết luồng liên kết rủi ro"):
                            st.write(
                                "Mỗi rủi ro được kết nối với biện pháp kiểm soát tương ứng và sự kiện phát hiện cụ thể thông qua quan hệ `[:MITIGATES]` và `[:OBSERVED_AS]`."
                            )
                except Exception as e:
                    st.error(f"❌ Đã xảy ra lỗi khi thực thi Graph RAG: {e}")
        else:
            st.warning("Chế độ Vector RAG đang chờ tích hợp thêm dữ liệu văn bản từ ChromaDB.")