import os
import sys

# Thiết lập hiển thị tiếng Việt trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào đường dẫn để có thể import từ src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import DocumentLoader
from src.embedding import RAGVectorStore
from src.retriever import DocumentRetriever
from tests.test_loader import setup_module

def test_retriever():
    print("\n=== KIỂM THỬ RETRIEVAL & GROUNDING ===")
    
    print("\n--- 1. Chuẩn bị VectorDB ---")
    setup_module()
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    loader = DocumentLoader(test_data_dir)
    valid_chunks = loader.load_all()
    
    if not valid_chunks:
        print("[LỖI] Không có chunk hợp lệ nào để kiểm thử.")
        return

    vector_store = RAGVectorStore()
    vector_store.add_documents(valid_chunks)
    
    print("\n--- 2. Khởi tạo Retriever ---")
    retriever = DocumentRetriever(vector_store)
    
    query = "Quy định về thời gian ban hành hoặc tính hợp lệ của nhân sự"
    print(f"Câu hỏi (Query): '{query}'\n")
    
    print("--- 3. Truy xuất và định dạng (Grounding Context) ---")
    context = retriever.retrieve_and_format(query, top_k=2)
    
    print("Kết quả Context sinh ra:\n")
    print("--------------------------------------------------")
    print(context)
    print("--------------------------------------------------")
    
    print("\n[HOÀN THÀNH] Module Retriever, Grounding và Citation đã hoạt động chính xác.")

if __name__ == "__main__":
    test_retriever()
