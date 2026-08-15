import unittest
from unittest.mock import patch
from advanced_rag import rerank_candidates, sigmoid

class TestRerank(unittest.TestCase):
    def setUp(self):
        self.fused = [
            {"chunk_id": "C1", "text": "Text 1", "fused_rank": 1},
            {"chunk_id": "C2", "text": "Text 2", "fused_rank": 2},
            {"chunk_id": "C3", "text": "Text 3", "fused_rank": 3},
            {"chunk_id": "C4", "text": "Text 4", "fused_rank": 4},
            {"chunk_id": "C5", "text": "Text 5", "fused_rank": 5},
            {"chunk_id": "C6", "text": "Text 6", "fused_rank": 6},
        ]

    def custom_fake_rerank(self, question, texts):
        # Fake scores to reverse the order (just for testing sort)
        # e.g., C6 gets highest score
        return [float(i) for i in range(len(texts))]

    @patch('advanced_rag.load_reranker_model')
    @patch('advanced_rag.load_advanced_config')
    def test_rerank_logic_and_limits(self, mock_config, mock_load):
        # 7. Chỉ rerank giới hạn candidate (e.g., 5)
        # 8. Chỉ trả final top-k (e.g., 3)
        mock_config.return_value = {
            'RERANK_CANDIDATES': 5,
            'FINAL_TOP_K': 3,
            'RERANK_BATCH_SIZE': 2,
            'RERANKER_MAX_LENGTH': 512,
            'RERANKER_MODEL': 'test-model'
        }
        
        # 10. Test không tải model (use custom_rerank_fn)
        # 2. Một pair cho mỗi candidate (custom fn receives list of texts)
        res = rerank_candidates("query", self.fused, custom_rerank_fn=self.custom_fake_rerank)
        
        results = res["results"]
        trace = res["trace"]
        
        # 1. Lazy loading (mock_load not called if custom_fn provided, meaning real model wasn't loaded)
        mock_load.assert_not_called()
        
        # RERANK_CANDIDATES was 5, so trace should show 5 reranked
        self.assertEqual(trace["rerank_candidate_count"], 5)
        # FINAL_TOP_K was 3, so results should have 3 items
        self.assertEqual(len(results), 3)
        
        # The fake scores for 5 items were [0.0, 1.0, 2.0, 3.0, 4.0]
        # Sigmoid of 4.0 > 3.0 > 2.0, so C5 comes first.
        self.assertEqual(results[0]["chunk_id"], "C5") # Highest score
        self.assertEqual(results[1]["chunk_id"], "C4")
        self.assertEqual(results[2]["chunk_id"], "C3")
        
        # 4. Sigmoid score đúng
        self.assertAlmostEqual(results[0]["rerank_score"], sigmoid(4.0))
        
        # 6. rank_change đúng (C5 fused_rank was 5, new rank is 1. change = 5 - 1 = 4)
        self.assertEqual(results[0]["rank_change"], 4) # 5 -> 1
        
    @patch('advanced_rag.load_advanced_config')
    def test_tie_break(self, mock_config):
        mock_config.return_value = {
            'RERANK_CANDIDATES': 2,
            'FINAL_TOP_K': 2,
            'RERANKER_MODEL': 'test-model'
        }
        
        fused = [
            {"chunk_id": "C2", "text": "T", "fused_rank": 1},
            {"chunk_id": "C1", "text": "T", "fused_rank": 2},
        ]
        
        # return same score for both
        def fake_same(q, texts): return [0.5, 0.5]
        
        res = rerank_candidates("query", fused, custom_rerank_fn=fake_same)
        results = res["results"]
        
        # 5. Sort và tie-break đúng
        # Scores are same. fused_rank 1 (C2) < fused_rank 2 (C1), so C2 wins tie break
        self.assertEqual(results[0]["chunk_id"], "C2")
        self.assertEqual(results[1]["chunk_id"], "C1")

    @patch('advanced_rag.load_reranker_model')
    @patch('advanced_rag.load_advanced_config')
    def test_model_failure_no_fallback(self, mock_config, mock_load):
        # 9. Model lỗi không silent fallback
        mock_config.return_value = {
            'RERANK_CANDIDATES': 5,
            'FINAL_TOP_K': 3
        }
        mock_load.side_effect = Exception("reranker_unavailable: Failed to load")
        
        with self.assertRaises(RuntimeError) as context:
            rerank_candidates("query", self.fused) # no custom fn -> trigger real model
            
        self.assertTrue("reranker_unavailable" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
