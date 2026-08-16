import os
import sys
import json
import unittest

# Thêm đường dẫn để có thể import từ src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import DocumentLoader
from src.retriever import DocumentRetriever
from src.generator import RAGGenerator

# Tạo class Mock Vector Store để kiểm thử độc lập không phụ thuộc ChromaDB/Windows DLL
class MockVectorStore:
    def search(self, query, n_results):
        return {
            'documents': [['Nội dung giả lập 1', 'Nội dung giả lập 2']],
            'metadatas': [
                [{'so_hieu': 'QD-01', 'tieu_de': 'Quy định 1'}, {'so_hieu': 'QD-02', 'tieu_de': 'Quy định 2'}]
            ],
            'distances': [[0.1, 0.2]]
        }

class TestRAGPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Khởi tạo môi trường ảo cho các Unit Test"""
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_data_unit")
        os.makedirs(cls.test_dir, exist_ok=True)
        
        cls.test_file = os.path.join(cls.test_dir, "sample.json")
        sample_data = [
            {"so_hieu": "QD-1", "tieu_de": "Chuẩn", "ngay_ban_hanh": "2026-01-01", "con_hieu_luc": True, "content": "Tốt"},
            {"so_hieu": "QD-2", "tieu_de": "", "ngay_ban_hanh": "2026-01-01", "con_hieu_luc": True, "content": "Lỗi tiêu đề rỗng"}
        ]
        
        with open(cls.test_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False)
            
    @classmethod
    def tearDownClass(cls):
        """Dọn dẹp môi trường sau khi hoàn thành test"""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)
        if os.path.exists(cls.test_dir):
            os.rmdir(cls.test_dir)

    def test_01_document_loader(self):
        """Kiểm thử việc nạp dữ liệu và Validate Pydantic"""
        loader = DocumentLoader(self.test_dir)
        chunks = loader.load_all()
        # Trong 2 bản ghi, 1 bản ghi lỗi tiêu đề rỗng nên chỉ còn 1 chunk
        self.assertEqual(len(chunks), 1, "Chỉ được phép nạp các dữ liệu hợp lệ (Pass validation).")
        self.assertEqual(chunks[0].metadata.so_hieu, "QD-1", "Bản ghi hợp lệ phải được giữ lại.")
        
    def test_02_retriever_formatting(self):
        """Kiểm thử định dạng Grounding & Citation của Retriever"""
        vector_store = MockVectorStore()
        retriever = DocumentRetriever(vector_store)
        
        context = retriever.retrieve_and_format("câu hỏi giả định", top_k=2)
        
        # Kiểm tra xem Header trích dẫn có đúng chuẩn không
        self.assertIn("--- Nguồn [QD-01]: Quy định 1 ---", context, "Định dạng Citation số 1 bị sai!")
        self.assertIn("--- Nguồn [QD-02]: Quy định 2 ---", context, "Định dạng Citation số 2 bị sai!")
        self.assertIn("Nội dung giả lập 1", context, "Không trích xuất đúng nội dung văn bản.")

    def test_03_generator_error_handling(self):
        """Kiểm thử cơ chế Catch Error của Generator khi API Key không hợp lệ"""
        # Nếu không có file .env chuẩn thì hệ thống sẽ văng lỗi auth
        generator = RAGGenerator(model_name="gemini-1.5-flash")
        
        context = "Đây là thông tin context"
        query = "Thử nghiệm"
        
        answer = generator.generate_answer(query, context)
        
        # Kiểm tra tính an toàn (Không crash ứng dụng, trả về thông báo lỗi 401 hoặc API)
        self.assertIn("[LỖI LLM GENERATION]", answer, "Cơ chế bắt lỗi của Generator chưa hoạt động đúng hoặc không trả về tiền tố [LỖI LLM GENERATION].")

if __name__ == '__main__':
    # Fix encode lỗi tiếng việt trên Windows cmd
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
        
    unittest.main(verbosity=2)
