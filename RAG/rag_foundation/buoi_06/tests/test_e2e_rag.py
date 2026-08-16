import os
import sys

# Đảm bảo UTF-8 để không lỗi tiếng Việt trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import DocumentLoader
from src.embedding import RAGVectorStore
from src.retriever import DocumentRetriever
from src.generator import RAGGenerator
from tests.test_loader import setup_module

def test_e2e_rag_pipeline():
    print("\n" + "="*50)
    print("      KIỂM THỬ TOÀN TRÌNH RAG PIPELINE (E2E)")
    print("="*50)
    
    # --- BƯỚC 1: PREPARE DATA ---
    print("\n[1/4] Chuẩn bị dữ liệu và Validator...")
    setup_module()
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    loader = DocumentLoader(test_data_dir)
    valid_chunks = loader.load_all()
    
    # --- BƯỚC 2: VECTOR DB ---
    print("\n[2/4] Nạp dữ liệu vào Vector DB (ChromaDB) / (hoặc Mock DB)...")
    vector_store = RAGVectorStore()
    vector_store.add_documents(valid_chunks)
    
    # --- BƯỚC 3: RETRIEVAL ---
    print("\n[3/4] Tìm kiếm ngữ nghĩa (Retrieval)...")
    retriever = DocumentRetriever(vector_store)
    
    # Câu hỏi thử nghiệm:
    queries = [
        "Tài liệu nào quy định về khoảng trắng thừa và nó ban hành ngày nào?",
        "Ngân hàng có hỗ trợ thai sản 12 tháng không?"  # Câu hỏi không có trong data để test tính trung thực (Chống ảo giác)
    ]
    
    generator = RAGGenerator(model_name="gemini-1.5-flash")
    
    # --- BƯỚC 4: GENERATION ---
    print("\n[4/4] Khởi tạo mô hình LLM và trả lời...\n")
    
    for i, query in enumerate(queries):
        print("-" * 50)
        print(f"HỎI (Q{i+1}): {query}")
        
        # Lấy context
        context = retriever.retrieve_and_format(query, top_k=2)
        print(f"\n[RETRIEVED CONTEXT]:\n{context}\n")
        
        # Sinh câu trả lời
        print(f"[GEMINI ĐANG SUY NGHĨ...]")
        answer = generator.generate_answer(query, context)
        
        print(f"\n[ĐÁP ÁN (A{i+1})]:\n{answer}")
        print("-" * 50)
        
    print("\n[HOÀN THÀNH] Đã chạy toàn bộ quy trình E2E RAG Pipeline!")

if __name__ == "__main__":
    test_e2e_rag_pipeline()
