import unittest
from unittest.mock import patch, MagicMock
from advanced_rag import rrf_fusion, hybrid_search

class TestHybrid(unittest.TestCase):
    def setUp(self):
        self.bm25_res = [
            {"chunk_id": "C1", "text": "T1", "source": "S1", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 10.0},
            {"chunk_id": "C2", "text": "T2", "source": "S2", "page_start": 1, "page_end": 1, "bm25_rank": 2, "bm25_score": 8.0}
        ]
        self.sem_res = [
            {"chunk_id": "C2", "text": "T2", "source": "S2", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1},
            {"chunk_id": "C3", "text": "T3", "source": "S3", "page_start": 1, "page_end": 1, "semantic_rank": 2, "semantic_distance": 0.2}
        ]

    def test_rrf_formula_and_overlap(self):
        # 1. RRF formula đúng toán học & 2. Overlap không duplicate
        # 3. Chỉ có BM25 được giữ & 4. Chỉ có semantic được giữ
        fused = rrf_fusion(self.bm25_res, self.sem_res, rrf_k=60, bm25_w=1.0, semantic_w=1.0)
        
        # 3 chunks total: C1, C2, C3
        self.assertEqual(len(fused), 3)
        
        # Expected C2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        c2 = next(x for x in fused if x["chunk_id"] == "C2")
        expected_c2 = 1.0/62 + 1.0/61
        self.assertAlmostEqual(c2["rrf_score"], expected_c2)
        self.assertCountEqual(c2["matched_by"], ["bm25", "semantic"])
        
        # Expected C1: 1/(60+1) = 1/61
        c1 = next(x for x in fused if x["chunk_id"] == "C1")
        self.assertAlmostEqual(c1["rrf_score"], 1.0/61)
        self.assertEqual(c1["matched_by"], ["bm25"])
        
        # Expected C3: 1/(60+2) = 1/62
        c3 = next(x for x in fused if x["chunk_id"] == "C3")
        self.assertAlmostEqual(c3["rrf_score"], 1.0/62)
        self.assertEqual(c3["matched_by"], ["semantic"])
        
        # Check fused rank
        self.assertEqual(fused[0]["chunk_id"], "C2")
        self.assertEqual(fused[1]["chunk_id"], "C1")
        self.assertEqual(fused[2]["chunk_id"], "C3")
        self.assertEqual(fused[0]["fused_rank"], 1)
        
    def test_weight_zero(self):
        # 5. Weight 0 loại đóng góp đúng nhánh
        fused = rrf_fusion(self.bm25_res, self.sem_res, rrf_k=60, bm25_w=0.0, semantic_w=1.0)
        c1 = next(x for x in fused if x["chunk_id"] == "C1")
        self.assertEqual(c1["rrf_score"], 0.0) # BM25 weight is 0
        
        c2 = next(x for x in fused if x["chunk_id"] == "C2")
        self.assertAlmostEqual(c2["rrf_score"], 1.0/61) # Only semantic

    def test_tie_break_deterministic(self):
        # 6. Tie-break deterministic.
        # Two chunks with identical rrf_score (e.g. both have rank 1 from different branches)
        b = [{"chunk_id": "C2", "text": "T2", "source": "S", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 10.0}]
        s = [{"chunk_id": "C1", "text": "T1", "source": "S", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}]
        
        fused = rrf_fusion(b, s, rrf_k=60, bm25_w=1.0, semantic_w=1.0)
        # score is 1/61 for both. 
        # tie break: best_rank is 1 for both.
        # sem_rank is 1 for C1, inf for C2.
        # C1 should come first!
        self.assertEqual(fused[0]["chunk_id"], "C1")
        self.assertEqual(fused[1]["chunk_id"], "C2")

    def test_metadata_mismatch_fails(self):
        # 7. Metadata mismatch fail
        sem_res_wrong = [{"chunk_id": "C2", "text": "WRONG_TEXT", "source": "S2", "page_start": 1, "page_end": 1, "semantic_rank": 1, "semantic_distance": 0.1}]
        with self.assertRaises(ValueError):
            rrf_fusion(self.bm25_res, sem_res_wrong, 60, 1.0, 1.0)

    @patch('advanced_rag.search_bm25')
    @patch('advanced_rag.search_semantic')
    @patch('rag.load_chunks')
    def test_hybrid_search_counts_and_calls(self, mock_load, mock_sem, mock_bm25):
        # 8. Trace counts đúng. & 9. Hybrid gọi mỗi retriever đúng một lần.
        mock_load.return_value = ([], {})
        mock_bm25.return_value = self.bm25_res
        mock_sem.return_value = self.sem_res
        
        # 10. Không load reranker/generation (by patching only retrievers, if it called others it would fail or we could check)
        res = hybrid_search("test query", "hierarchical")
        
        mock_bm25.assert_called_once()
        mock_sem.assert_called_once()
        
        trace = res["trace"]
        self.assertEqual(trace["bm25_candidate_count"], 2)
        self.assertEqual(trace["semantic_candidate_count"], 2)
        self.assertEqual(trace["union_count"], 3)
        self.assertEqual(trace["overlap_count"], 1)
        self.assertEqual(trace["fused_count"], 3)
        self.assertIn("bm25", trace["latency_ms"])

if __name__ == '__main__':
    unittest.main()
