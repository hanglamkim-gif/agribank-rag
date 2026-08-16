import os
import sys

# Thêm thư mục gốc vào đường dẫn để có thể import từ src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import DocumentLoader
from src.embedding import RAGVectorStore
from tests.test_loader import setup_module, teardown_module

def test_chroma_integration():
    print("\n--- 1. NẠP DỮ LIỆU TỪ LOADER ---")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..") # trỏ ra thư mục gốc c:\agribank-rag
    
    # Tạo lại test data
    setup_module()
    
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    
    loader = DocumentLoader(test_data_dir)
    valid_chunks = loader.load_all()
    
    if not valid_chunks:
        print("[LỖI] Không có chunk hợp lệ nào để nạp.")
        return

    print(f"Tổng số chunk hợp lệ để nạp: {len(valid_chunks)}")
    
    print("\n--- 2. KHỞI TẠO VÀ NẠP VÀO CHROMADB ---")
    vector_store = RAGVectorStore()
    vector_store.add_documents(valid_chunks)
    
    print("\n--- 3. TRUY VẤN TÌM KIẾM NGỮ NGHĨA (SEMANTIC SEARCH) ---")
    query = "Tài liệu nào liên quan đến nội dung hợp lệ?"
    print(f"Query: '{query}'")
    
    results = vector_store.search(query, n_results=2)
    
    print("\n[KẾT QUẢ TÌM KIẾM]")
    for idx, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"\nKết quả {idx + 1} (Distance: {dist:.4f})")
        print(f" - Số hiệu: {meta.get('so_hieu')}")
        print(f" - Tiêu đề: {meta.get('tieu_de')}")
        print(f" - Nội dung: {doc}")
        
    print("\n[HOÀN THÀNH] ChromaDB Index đã hoạt động trơn tru.")

if __name__ == "__main__":
    test_chroma_integration()
