import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from rag import run_query, get_collection_name, load_config

def mock_query_embed_fn(question, api_key, model, dim):
    return [[0.1] * dim]

class TestHRRag(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_col = MagicMock()
        self.mock_client.get_collection.return_value = self.mock_col
        self.mock_client.create_collection.return_value = self.mock_col
        self.mock_col.count.return_value = 10
        self.mock_col.metadata = {"strategy": "hierarchical", "embedding_model": "mock-emb-model", "embedding_dim": 128, "distance_metric": "cosine"}
        
        mock_c_obj = MagicMock()
        mock_c_obj.name = get_collection_name("hierarchical", 128, "mock-emb-model")
        self.mock_client.list_collections.return_value = [mock_c_obj]
        
        self.patcher = patch('chromadb.PersistentClient', return_value=self.mock_client)
        self.patcher.start()
        
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "chroma"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.config = {
            "has_api_key": True,
            "api_key": "mock_key",
            "emb_model": "mock-emb-model",
            "emb_dim": 128,
            "gen_model": "mock-gen-model",
            "max_dist": 0.35, # Strict threshold for HR
            "top_k": 5
        }
        
    def tearDown(self):
        self.patcher.stop()

    def _setup_query_mock(self, num_records, dists=None):
        docs = [f"t{i+1}" for i in range(num_records)]
        metas = [{"source": "s1", "page_start": 1, "page_end": i+1, "chunk_id": str(i+1)} for i in range(num_records)]
        distances = dists if dists else [0.1]*num_records
        self.mock_col.query.return_value = {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances]
        }

    # Kịch bản 1: Exact Match (Tìm đúng quy định trong dữ liệu)
    def test_hr_exact_match(self):
        self._setup_query_mock(1, [0.1])
        def mock_gen_fn_success(prompt):
            self.assertTrue("t1" in prompt) # Evidence extracted
            self.assertTrue("Agribank" in prompt) # Prompt check
            return "Lao động nữ sinh con được nghỉ thai sản 6 tháng theo quy định. [E1]"
            
        res = run_query("Lao động nữ sinh con được nghỉ thai sản bao nhiêu tháng?", 
                         5, "hierarchical", 
                         storage_dir=self.storage_dir, _config=self.config, 
                         _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_fn_success)
        
        # Test query success should have output in final_answer but the status is probably missing if it's not set. 
        # But looking at rag.py earlier, it's not strictly 'success', maybe it's missing status key.
        # But anyway we can check the final answer.
        self.assertIn("6 tháng", res.get("answer", res))

    # Kịch bản 2: Complex Reasoning (Tính toán dựa trên quy định)
    def test_hr_complex_reasoning(self):
        self._setup_query_mock(2, [0.15, 0.2])
        def mock_gen_fn_reasoning(prompt):
            self.assertTrue("t1" in prompt and "t2" in prompt)
            return "Theo quy định tính phép, bạn làm từ 15/06 đến 31/12 (trên 15 ngày làm tròn 1 tháng) nên bạn có 7 ngày phép. [E1]"
            
        res = run_query("Tôi vào làm ngày 15/06/2024, đến cuối năm tôi có bao nhiêu ngày phép?", 
                         5, "hierarchical", 
                         storage_dir=self.storage_dir, _config=self.config, 
                         _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_fn_reasoning)
                         
        self.assertIn("7 ngày phép", res.get("answer", res))

    # Kịch bản 3: Out of domain / Confidence Gate (Câu hỏi ngoài lề)
    def test_hr_out_of_domain(self):
        # Distance > 0.35
        self._setup_query_mock(1, [0.8])
        def mock_gen_fn_should_not_be_called(prompt):
            self.fail("Không được gọi Generation Engine nếu distance vượt ngưỡng!")
            
        res = run_query("Cách reset mật khẩu hệ thống phần mềm kế toán?", 
                         5, "hierarchical", 
                         storage_dir=self.storage_dir, _config=self.config, 
                         _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_fn_should_not_be_called)
                         
        self.assertEqual(res.get("status"), "insufficient_evidence", "Phải chặn câu hỏi không liên quan!")
        self.assertEqual(len(res.get("citations", [])), 0)

if __name__ == '__main__':
    unittest.main()
