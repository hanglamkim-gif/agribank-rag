import unittest
from unittest.mock import patch, MagicMock
import os
import json
from advanced_rag import search_semantic, show_status, prepare_semantic, load_advanced_config

class TestSemantic(unittest.TestCase):
    def setUp(self):
        # Mock .env for api key and other configs
        self.patcher_env = patch.dict(os.environ, {
            "GEMINI_API_KEY": "test_key",
            "GEMINI_EMBEDDING_MODEL": "test-embed",
            "GEMINI_EMBEDDING_DIM": "768",
            "RERANKER_MODEL": "test-reranker"
        })
        self.patcher_env.start()
        
        self.mock_client = MagicMock()
        self.mock_col = MagicMock()
        self.mock_client.get_collection.return_value = self.mock_col
        self.mock_col.metadata = {"strategy": "hierarchical"}
        
        # Mock query return
        self.mock_col.count.return_value = 2
        self.mock_col.query.return_value = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"chunk_id": "C1", "source": "S1", "page_start": 1, "page_end": 1},
                           {"chunk_id": "C2", "source": "S2", "page_start": 2, "page_end": 2}]],
            "distances": [[0.1, 0.2]]
        }
        
        self.patcher_client = patch('chromadb.PersistentClient', return_value=self.mock_client)
        self.patcher_client.start()
        
        self.patcher_embed = patch('rag.get_embedding_gemini', return_value=[0.1]*768)
        self.patcher_embed.start()
        
    def tearDown(self):
        self.patcher_env.stop()
        self.patcher_client.stop()
        self.patcher_embed.stop()
        
    def test_semantic_search_success(self):
        # 1. semantic top-k/count/order đúng & 2. metadata đầy đủ & 6. không gọi generation
        results = search_semantic("test question", 10, "hierarchical")
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "C1")
        self.assertEqual(results[0]["semantic_rank"], 1)
        self.assertEqual(results[0]["semantic_distance"], 0.1)
        
        self.assertEqual(results[1]["chunk_id"], "C2")
        self.assertEqual(results[1]["source"], "S2")
        self.assertEqual(results[1]["semantic_rank"], 2)
        
    def test_collection_mismatch_blocked(self):
        # 3. collection mismatch bị chặn
        self.mock_col.metadata = {"strategy": "fixed-size"}
        with self.assertRaises(ValueError):
            search_semantic("test", 10, "hierarchical")
            
    def test_status_no_collection_creation(self):
        # 4. status không tạo collection
        self.mock_client.get_collection.side_effect = Exception("Not found")
        # Should run without error and without creating collection
        show_status("hierarchical")
        self.mock_client.create_collection.assert_not_called()
        
    @patch.dict(os.environ, {"GEMINI_API_KEY": ""})
    def test_missing_api_key_fails(self):
        # 5. không có key không dùng vector giả
        with self.assertRaises(ValueError):
            prepare_semantic("hierarchical")
            
        with self.assertRaises(ValueError):
            search_semantic("test", 10, "hierarchical")

if __name__ == '__main__':
    unittest.main()
