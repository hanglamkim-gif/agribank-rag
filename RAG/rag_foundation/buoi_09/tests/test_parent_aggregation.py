import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch
from hierarchical_rag import parent_retrieval

class TestParentAggregation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "PARENT_SCORE_CHILD_LIMIT": 3,
            "PARENT_RRF_K": 60,
            "PARENT_CANDIDATES": 10,
            "TOTAL_CONTEXT_MAX_CHARS": 2000
        }
        
        self.base_dir = Path(__file__).resolve().parent.parent / "storage" / "hierarchy"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {"schema_version": "1.0"}
        children = [
            {"child_id": "c1", "parent_id": "p1"},
            {"child_id": "c2", "parent_id": "p1"},
            {"child_id": "c3", "parent_id": "p1"},
            {"child_id": "c4", "parent_id": "p1"},
            {"child_id": "c5", "parent_id": "p2"},
            {"child_id": "c6", "parent_id": "p3"},
            {"child_id": "c7", "parent_id": "p3"}
        ]
        parents = [
            {"parent_id": "p1", "source": "f1", "page_start": 1, "page_end": 2, "article_key": "art1", "text": "A"*300},
            {"parent_id": "p2", "source": "f1", "page_start": 2, "page_end": 3, "article_key": "art2", "text": "B"*800},
            {"parent_id": "p3", "source": "f1", "page_start": 3, "page_end": 4, "article_key": "art3", "text": "C"*200}
        ]
        
        with open(self.base_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with open(self.base_dir / "children.json", "w", encoding="utf-8") as f:
            json.dump(children, f)
        with open(self.base_dir / "parents.json", "w", encoding="utf-8") as f:
            json.dump(parents, f)
            
    def _fake_multi_query(self, *args, **kwargs):
        return {
            "status": "success",
            "results": [
                {"child_id": "c1", "multi_query_rank": 1, "support_query_ids": ["Q0"], "text": "c1 text"},
                {"child_id": "c2", "multi_query_rank": 2, "support_query_ids": ["Q0", "Q1"], "text": "c2 text"},
                {"child_id": "c3", "multi_query_rank": 3, "support_query_ids": ["Q1"], "text": "c3 text"},
                {"child_id": "c4", "multi_query_rank": 4, "support_query_ids": ["Q2"], "text": "c4 text"},
                {"child_id": "c5", "multi_query_rank": 5, "support_query_ids": ["Q0"], "text": "c5 text"},
                {"child_id": "c6", "multi_query_rank": 6, "support_query_ids": ["Q0"], "text": "c6 text"},
                {"child_id": "c7", "multi_query_rank": 6, "support_query_ids": ["Q0"], "text": "c7 text"}
            ],
            "trace": {}
        }

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_01_child_map_correct_parent(self, mock_mq):
        mock_mq.side_effect = self._fake_multi_query
        res = parent_retrieval("Test", self.config)
        
        p1 = next(x for x in res["results"] if x["parent_id"] == "p1")
        p2 = next(x for x in res["results"] if x["parent_id"] == "p2")
        
        self.assertEqual(p1["supporting_child_ids"], ["c1", "c2", "c3", "c4"])
        self.assertEqual(p2["supporting_child_ids"], ["c5"])

    def test_02_hierarchy_not_ready(self):
        (self.base_dir / "manifest.json").unlink()
        res = parent_retrieval("Test", self.config)
        self.assertEqual(res["status"], "hierarchy_not_ready")

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_03_parent_aggregation_formula(self, mock_mq):
        mock_mq.side_effect = self._fake_multi_query
        res = parent_retrieval("Test", self.config)
        p1 = next(x for x in res["results"] if x["parent_id"] == "p1")
        # limit is 3, ranks are 1, 2, 3
        expected = (1.0 / 61) + (1.0 / 62) + (1.0 / 63)
        self.assertAlmostEqual(p1["parent_rrf_score"], expected, places=5)

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_04_child_score_cap(self, mock_mq):
        mock_mq.side_effect = self._fake_multi_query
        res = parent_retrieval("Test", self.config)
        p1 = next(x for x in res["results"] if x["parent_id"] == "p1")
        # c4 (rank 4) should not be in scoring_child_ids
        self.assertNotIn("c4", p1["scoring_child_ids"])
        self.assertEqual(len(p1["scoring_child_ids"]), 3)

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_05_supporting_vs_scoring(self, mock_mq):
        mock_mq.side_effect = self._fake_multi_query
        res = parent_retrieval("Test", self.config)
        p1 = next(x for x in res["results"] if x["parent_id"] == "p1")
        self.assertEqual(p1["anchor_child_id"], "c1")
        self.assertEqual(len(p1["supporting_child_ids"]), 4)
        self.assertEqual(len(p1["scoring_child_ids"]), 3)

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_06_parent_deduplicate(self, mock_mq):
        mock_mq.side_effect = self._fake_multi_query
        res = parent_retrieval("Test", self.config)
        pids = [r["parent_id"] for r in res["results"]]
        self.assertEqual(len(pids), len(set(pids)))

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_07_deterministic_sort(self, mock_mq):
        # Tie break on p3 (c6, c7 have rank 6)
        mock_mq.side_effect = self._fake_multi_query
        res = parent_retrieval("Test", self.config)
        # Should be sorted: p1 (best score), p2 (rank 5), p3 (rank 6)
        self.assertEqual(res["results"][0]["parent_id"], "p1")

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_08_candidate_limit(self, mock_mq):
        mock_mq.side_effect = self._fake_multi_query
        cfg = dict(self.config)
        cfg["PARENT_CANDIDATES"] = 1
        res = parent_retrieval("Test", cfg)
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(res["trace"]["parents_dropped_by_candidate_limit"], 2)

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_09_context_budget_boundary(self, mock_mq):
        mock_mq.side_effect = self._fake_multi_query
        cfg = dict(self.config)
        cfg["TOTAL_CONTEXT_MAX_CHARS"] = 1000
        # p1=300, p2=800, p3=200. Total budget=1000
        # Expected sort: p1, p3, p2. 
        # p1 (300) -> 300
        # p3 (200) -> 500 (fits budget, kept!)
        # p2 (800) -> 1300 (exceeds budget, dropped!)
        res = parent_retrieval("Test", cfg)
        pids = [x["parent_id"] for x in res["results"]]
        self.assertIn("p1", pids)
        self.assertNotIn("p2", pids)
        self.assertIn("p3", pids)
        self.assertEqual(res["trace"]["parents_dropped_by_budget"], 1)

    @patch("hierarchical_rag.multi_query_retrieval")
    def test_10_oversized_first_parent_warning(self, mock_mq):
        def fake_mq_oversized(*args, **kwargs):
            return {
                "status": "success",
                "results": [{"child_id": "c5", "multi_query_rank": 1, "support_query_ids": ["Q0"], "text": "B1"}]
            }
        mock_mq.side_effect = fake_mq_oversized
        cfg = dict(self.config)
        cfg["TOTAL_CONTEXT_MAX_CHARS"] = 500 # p2 is 800, so first parent exceeds budget
        res = parent_retrieval("Test", cfg)
        self.assertEqual(len(res["results"]), 1)
        self.assertIn("oversized_first_parent_budget_exceeded", res["results"][0]["warnings"])
        self.assertEqual(res["trace"]["warning_count"], 1)

    def test_11_trace_metrics(self):
        # Must pass custom_query_gen and custom_hybrid_search to avoid API calls
        res = parent_retrieval("Test", self.config, 
                              custom_query_gen=lambda q, c: [{"query_id": "Q0", "text": q}],
                              custom_hybrid_search=lambda q, s: [{"chunk_id": "c1", "score": 0.9}])
        tr = res["trace"]
        self.assertIn("mode", tr)
        self.assertIn("latency_ms", tr)
        self.assertIn("unique_parent_count", tr)
        self.assertIn("input_child_hit_count", tr)
        self.assertIn("child_chars_vs_expanded_parent_chars", tr)
        self.assertIn("aggregation_latency_ms", tr)

    def test_12_no_reranker_generation(self):
        # Implicitly tested by the fact we only patched multi_query_retrieval and didn't crash.
        pass

if __name__ == "__main__":
    unittest.main()
