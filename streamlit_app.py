import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Cấu hình giao diện trang web
st.set_page_config(page_title="Agribank RAG Assistant", page_icon="🤖", layout="centered")

st.title("🏦 Agribank - Trợ lý RAG Đồ thị & Vector")
st.write("Hệ thống tra cứu văn bản pháp luật tích hợp Neo4j và mô hình AI tiếng Việt.")

# 1. Tải mô hình Embedding (Có cache để chạy nhanh hơn)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5')

with st.spinner("Đang tải mô hình AI tiếng Việt..."):
    model = load_embedding_model()

# 2. Kết nối Neo4j với mật khẩu của bạn
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "Kh@141086")

@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(URI, auth=AUTH)

try:
    driver = get_neo4j_driver()
    driver.verify_connectivity()
    st.success("Kết nối cơ sở dữ liệu Neo4j thành công!")
except Exception as e:
    st.error(f"Không thể kết nối đến Neo4j. Hãy kiểm tra lại Neo4j Desktop đang chạy: {e}")

# 3. Khu vực nhập câu hỏi tìm kiếm
query_text = st.text_input("Nhập câu hỏi của bạn về quy định ngân hàng:", "Quy định về phạm vi điều chỉnh")

if st.button("🔍 Tìm kiếm thông tin"):
    if query_text.strip():
        with st.spinner("Đang xử lý vector hóa và truy vấn đồ thị..."):
            # Tạo vector nhúng cho câu hỏi
            query_vector = model.encode(query_text).tolist()

            # Truy vấn dữ liệu từ Neo4j
            with driver.session(database="neo4j") as session:
                result = session.run("""
                MATCH (c:Chunk)
                RETURN c.id AS id, c.type AS type, c.content AS content
                """)
                chunks = [{"id": record["id"], "type": record["type"], "content": record["content"]} for record in result]

            st.subheader("📌 Kết quả tra cứu:")
            if chunks:
                for i, chunk in enumerate(chunks):
                    with st.expander(f"Kết quả {i+1} - Loại: {chunk['type']} (ID: {chunk['id']})"):
                        st.write(f"**Nội dung:** {chunk['content']}")
            else:
                st.warning("Không tìm thấy dữ liệu phù hợp trong cơ sở dữ liệu.")
    else:
        st.warning("Vui lòng nhập nội dung câu hỏi.")