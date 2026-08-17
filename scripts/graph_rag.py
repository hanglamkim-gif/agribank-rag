import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from google import genai

# 1. Đọc cấu hình từ file .env ở thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_graph_context():
    """Truy vấn đồ thị Neo4j để lấy bối cảnh rủi ro, kiểm soát và sự kiện biểu hiện."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    context_data = []
    
    with driver.session(database=NEO4J_DATABASE) as session:
        query = """
            MATCH (k:KiemSoat)-[r1:MITIGATES]->(rr:RuiRo)-[r2:OBSERVED_AS]->(sk:SuKienRuiRo)
            RETURN k.name AS kiem_soat, k.description AS k_desc, 
                   rr.name AS rui_ro, rr.description AS rr_desc, 
                   sk.name AS su_kien, sk.description AS sk_desc
        """
        result = session.run(query)
        for record in result:
            context_data.append(
                f"- [Rủi ro]: {record['rui_ro']} (Chi tiết: {record['rr_desc']})\n"
                f"  + [Kiểm soát]: {record['kiem_soat']} (Chi tiết: {record['k_desc']})\n"
                f"  + [Sự kiện biểu hiện]: {record['su_kien']} (Chi tiết: {record['sk_desc']})"
            )
            
    driver.close()
    return "\n\n".join(context_data)


def answer_with_graph_rag(user_question):
    """
    Hàm xử lý Graph RAG:
    - Truy xuất ngữ cảnh từ Neo4j
    - Gửi ngữ cảnh + câu hỏi cho Gemini
    - Trả về tuple: (câu trả lời của AI, dữ liệu ngữ cảnh đồ thị)
    """
    # 1. Lấy ngữ cảnh đồ thị
    context = get_graph_context()
    
    # 2. Xây dựng prompt nghiệp vụ
    prompt = f"""
Bạn là một Trợ lý AI chuyên gia về Quản trị Rủi ro và Kiểm soát Nội bộ tại Agribank.
Dựa trên cấu trúc đồ thị tri thức rủi ro dưới đây, hãy phân tích câu hỏi nghiệp vụ của người dùng một cách chuyên nghiệp, chi tiết và có cấu trúc rõ ràng:

--- CẤU TRÚC ĐỒ THỊ TRI THỨC (GRAPH CONTEXT) ---
{context}

--- CÂU HỎI NGHIỆP VỤ ---
{user_question}
"""
    # 3. Gọi Gemini API
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt
    )
    
    # Trả về cả câu trả lời và context để Streamlit hiển thị
    return response.text, context


# 4. Khối thực thi khi chạy trực tiếp từ Terminal
if __name__ == "__main__":
    cau_hoi = "Hãy phân tích rủi ro hạch toán sai giao dịch chuyển tiền và các biện pháp kiểm soát tương ứng."
    print("1. Đang truy xuất tri thức từ Neo4j...")
    answer, graph_context = answer_with_graph_rag(cau_hoi)
    
    print("\n--- [DỮ LIỆU ĐỒ THỊ TRÍCH XUẤT ĐƯỢC] ---")
    print(graph_context)
    print("------------------------------------------\n")
    
    print("=== KẾT QUẢ PHÂN TÍCH GRAPH RAG ===")
    print(answer)