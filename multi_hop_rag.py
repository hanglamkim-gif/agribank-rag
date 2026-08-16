import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from google import genai
import os

# --- CẤU HÌNH KẾT NỐI NEO4J ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "Kh@141086")
DATABASE_NAME = "kb-hops"

# Khởi tạo mô hình Embedding tiếng Việt
@st.cache_resource
def load_model():
    return SentenceTransformer('thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5')

model = load_model()

# Khởi tạo Gemini Client chuẩn
client = genai.Client()

# --- HÀM TRUY VẤN MULTI-HOP GRAPH RAG ---
def search_multi_hop_graph(query_text, hops=1):
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    cypher_query = f"""
    MATCH (c:Chunk)
    MATCH path = (c)-[:PARENT_OF|PART_OF|NEXT*0..{hops}]-(related)
    RETURN DISTINCT related.content AS content, related.type AS type, related.id AS id
    LIMIT 5
    """
    
    contexts = []
    try:
        with driver.session(database=DATABASE_NAME) as session:
            result = session.run(cypher_query)
            for record in result:
                if record["content"]:
                    contexts.append(f"[{record['type']}] {record['content']}")
    except Exception:
        with driver.session(database="neo4j") as session:
            result = session.run(cypher_query)
            for record in result:
                if record["content"]:
                    contexts.append(f"[{record['type']}] {record['content']}")
    finally:
        driver.close()
        
    return "\n".join(contexts)

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="Multi-hop Graph RAG Assistant", page_icon="🤖")
st.title("🔎 Multi-hop Graph RAG & QA (Bài 11)")

question = st.text_input("Nhập câu hỏi kiểm thử:", "Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào?")
hops_count = st.slider("Cấu hình số bước nhảy (Multi-hop hops):", min_value=0, max_value=2, value=1)

if st.button("Thực hiện truy vấn và hỏi LLM"):
    if question.strip():
        with st.spinner("Đang trích xuất ngữ cảnh đa bước từ Neo4j..."):
            context_data = search_multi_hop_graph(question, hops=hops_count)
            
        with st.spinner("Đang gọi Gemini sinh câu trả lời..."):
            prompt = f"""
            Bạn là một trợ lý pháp lý AI chuyên nghiệp. Hãy sử dụng thông tin ngữ cảnh từ đồ thị tri thức dưới đây để trả lời câu hỏi một cách chính xác. Nếu không có thông tin, hãy nói rõ là không biết.

            --- NGỮ CẢNH ĐỒ THỊ ({hops_count} bước nhảy) ---
            {context_data}
            ------------------------------------------------
            
            Câu hỏi: {question}
            Trả lời:
            """
            
            # Sử dụng model định danh mới nhất gemini-3.5-flash
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt
            )
            
            st.subheader("📌 Câu trả lời từ AI:")
            st.write(response.text)
            
            with st.expander("Xem chi tiết ngữ cảnh đồ thị đã thu thập:"):
                st.text(context_data)
    else:
        st.warning("Vui lòng nhập câu hỏi.")