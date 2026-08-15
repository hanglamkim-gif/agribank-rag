import unittest
from unittest.mock import patch, MagicMock
from advanced_rag import advanced_generate_answer, run_compare

class TestAnswer(unittest.TestCase):
    def setUp(self):
        # Fake chunks list returned from mock
        self.fake_chunks = [
            {"chunk_id": "C1", "text": "T1", "source": "S1", "page_start": 1, "page_end": 1},
            {"chunk_id": "C2", "text": "T2", "source": "S2", "page_start": 2, "page_end": 2}
        ]
        
        self.fake_bm25_res = [
            {"chunk_id": "C1", "text": "T1", "source": "S1", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 10.0},
            {"chunk_id": "C2", "text": "T2", "source": "S2", "page_start": 2, "page_end": 2, "bm25_rank": 2, "bm25_score": 8.0}
        ]
        
        self.fake_sem_res = [
            {"chunk_id": "C1", "text": "T1", "source": "S1", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.2},
            {"chunk_id": "C2", "text": "T2", "source": "S2", "page_start": 2, "page_end": 2, "semantic_rank": 2, "semantic_distance": 0.6} # 0.6 > max_dist(0.45)
        ]
        
        self.fake_hybrid_res = {
            "results": [
                {"chunk_id": "C1", "text": "T1", "source": "S1", "page_start": 1, "page_end": 1, "bm25_rank": 1, "semantic_rank": 1, "semantic_distance": 0.2, "rrf_score": 0.05, "fused_rank": 1, "matched_by": ["bm25", "semantic"]},
                {"chunk_id": "C2", "text": "T2", "source": "S2", "page_start": 2, "page_end": 2, "bm25_rank": 2, "semantic_rank": 2, "semantic_distance": 0.6, "rrf_score": 0.03, "fused_rank": 2, "matched_by": ["bm25", "semantic"]}
            ],
            "trace": {
                "bm25_candidate_count": 2, "semantic_candidate_count": 2, "overlap_count": 2, "union_count": 2,
                "latency_ms": {"bm25": 10, "semantic": 10, "fusion": 5}
            }
        }
        
        self.fake_rerank_res = {
            "results": [
                {"chunk_id": "C1", "text": "T1", "source": "S1", "page_start": 1, "page_end": 1, "bm25_rank": 1, "semantic_rank": 1, "semantic_distance": 0.2, "rrf_score": 0.05, "fused_rank": 1, "matched_by": ["bm25", "semantic"], "rerank_score": 0.9, "rerank_rank": 1, "rank_change": 0},
                {"chunk_id": "C2", "text": "T2", "source": "S2", "page_start": 2, "page_end": 2, "bm25_rank": 2, "semantic_rank": 2, "semantic_distance": 0.6, "rrf_score": 0.03, "fused_rank": 2, "matched_by": ["bm25", "semantic"], "rerank_score": 0.3, "rerank_rank": 2, "rank_change": 0} # 0.3 < min(0.5)
            ],
            "trace": {"rerank_candidate_count": 2, "latency_ms": 50, "reranker_model": "test"}
        }

    @patch('advanced_rag.search_bm25')
    @patch('advanced_rag.search_semantic')
    @patch('advanced_rag.hybrid_search')
    @patch('advanced_rag.rerank_candidates')
    @patch('rag.load_chunks')
    def test_gating_and_rejected_evidence(self, mock_load, mock_rerank, mock_hybrid, mock_sem, mock_bm25):
        # Test hybrid_rerank gating (C1 passed, C2 rejected)
        mock_load.return_value = (self.fake_chunks, {})
        mock_hybrid.return_value = self.fake_hybrid_res
        mock_rerank.return_value = self.fake_rerank_res
        
        def fake_gen(prompt):
            self.assertIn("T1", prompt)
            self.assertNotIn("T2", prompt) # rejected evidence must not be in prompt
            return "Answer is here [E1]."
            
        res = advanced_generate_answer("Q", "hierarchical", mode="hybrid_rerank", custom_gen_fn=fake_gen)
        
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["trace"]["accepted"], 1)
        self.assertEqual(len(res["evidence"]), 2)
        
        # Verify accepted flags
        e1 = next(x for x in res["evidence"] if x["chunk_id"] == "C1")
        e2 = next(x for x in res["evidence"] if x["chunk_id"] == "C2")
        self.assertTrue(e1["accepted"])
        self.assertFalse(e2["accepted"])
        
        # 4. Citation maps to real metadata
        self.assertEqual(len(res["citations"]), 1)
        self.assertEqual(res["citations"][0]["chunk_id"], "C1")
        self.assertEqual(res["citations"][0]["label"], "[E1]")
        
    @patch('advanced_rag.hybrid_search')
    @patch('advanced_rag.rerank_candidates')
    def test_reranker_unavailable_status(self, mock_rerank, mock_hybrid):
        mock_hybrid.return_value = self.fake_hybrid_res
        mock_rerank.side_effect = Exception("reranker_unavailable: Failed to load")
        
        res = advanced_generate_answer("Q", "hierarchical", mode="hybrid_rerank")
        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertEqual(res["trace"]["generation_called"], False)

    @patch('advanced_rag.advanced_generate_answer')
    def test_compare_no_generation(self, mock_adv_gen):
        # 6. Compare không gọi generation (by passing custom_gen_fn that returns "skipped")
        mock_adv_gen.return_value = {
            "evidence": self.fake_bm25_res,
            "trace": {"latency_ms": {"total": 10, "bm25": 10, "semantic": 0, "fusion": 0, "rerank": 0}}
        }
        
        run_compare("Q", "hierarchical")
        
        # Called 4 times (for 4 modes)
        self.assertEqual(mock_adv_gen.call_count, 4)
        
        # Check that it passes a custom_gen_fn to skip generation
        call_args = mock_adv_gen.call_args_list[0][1]
        self.assertIn("custom_gen_fn", call_args)
        self.assertEqual(call_args["custom_gen_fn"](None), "skipped")

if __name__ == '__main__':
    unittest.main()
