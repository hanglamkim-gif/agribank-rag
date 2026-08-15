import unittest
import json
import hashlib
from hierarchical_rag import generate_query_variants, _QUERY_CACHE

class TestExpandQuery(unittest.TestCase):
    def setUp(self):
        self.config = {
            "MULTI_QUERY_COUNT": 3,
            "MULTI_QUERY_MAX_CHARS": 100,
            "MULTI_QUERY_TEMPERATURE": 0.2,
            "GENERATION_MODEL": "test-model"
        }
        _QUERY_CACHE.clear()

    def test_01_q0_always_first_and_preserved(self):
        def fake_gen(q, n):
            return [{"text": "B", "focus": "p"}]
            
        res = generate_query_variants("  A   \n B  ", self.config, fake_gen)
        self.assertEqual(res["queries"][0]["query_id"], "Q0")
        self.assertEqual(res["queries"][0]["text"], "A B")
        self.assertEqual(res["queries"][0]["origin"], "original")
        
    def test_02_strict_schema_validation(self):
        # Fake generator returning wrong structure throws exception or is caught
        def fake_gen(q, n):
            raise ValueError("Bad JSON")
            
        res = generate_query_variants("A", self.config, fake_gen)
        self.assertEqual(res["status"], "query_generation_unavailable")
        self.assertEqual(len(res["queries"]), 1)
        self.assertEqual(res["queries"][0]["query_id"], "Q0")
        
    def test_03_nfc_trim_max_length(self):
        def fake_gen(q, n):
            return [
                {"text": "   TRIM_ME   ", "focus": "p"},
                {"text": "A" * 150, "focus": "p"} # Exceeds max length (100)
            ]
        res = generate_query_variants("A", self.config, fake_gen)
        self.assertEqual(len(res["queries"]), 2) # Q0 + TRIM_ME
        self.assertEqual(res["queries"][1]["text"], "TRIM_ME")
        
    def test_04_duplicate_removal(self):
        def fake_gen(q, n):
            return [
                {"text": "B", "focus": "p"},
                {"text": "b", "focus": "p"}, # Duplicate of B (casefold)
                {"text": "B ", "focus": "p"} # Duplicate of B (whitespace)
            ]
        res = generate_query_variants("A", self.config, fake_gen)
        self.assertEqual(len(res["queries"]), 2) # Q0 + B
        self.assertEqual(res["dropped_duplicate_count"], 2)
        
    def test_05_legal_reference_preservation(self):
        # Q0 has "Điều 5"
        def fake_gen(q, n):
            return [
                {"text": "Điều 5 khoản 2", "focus": "p"},
                {"text": "Quy định abc", "focus": "p"}
            ]
        res = generate_query_variants("Thế nào là Điều 5?", self.config, fake_gen)
        self.assertEqual(len(res["queries"]), 3) # Both accepted
        
        # If generator fails to preserve "Điều 5"
        def fake_gen_bad(q, n):
            return [{"text": "Luật dân sự", "focus": "p"}]
        res_bad = generate_query_variants("Điều 6 là gì?", self.config, fake_gen_bad)
        self.assertEqual(len(res_bad["queries"]), 1) # Variants dropped
        
    def test_06_no_fabricated_references(self):
        # Q0 doesn't have Dieu, generator invents one
        def fake_gen(q, n):
            return [{"text": "Điều 10 luật", "focus": "p"}]
        res = generate_query_variants("Vay vốn", self.config, fake_gen)
        self.assertEqual(len(res["queries"]), 1) # Invented ref dropped
        
    def test_07_deterministic_ids(self):
        def fake_gen(q, n):
            return [{"text": "X", "focus": "p"}, {"text": "Y", "focus": "p"}]
        res = generate_query_variants("A", self.config, fake_gen)
        self.assertEqual(res["queries"][1]["query_id"], "Q1")
        self.assertEqual(res["queries"][2]["query_id"], "Q2")
        
    def test_08_one_generator_call(self):
        call_count = 0
        def fake_gen(q, n):
            nonlocal call_count
            call_count += 1
            return [{"text": "B", "focus": "p"}]
        generate_query_variants("A", self.config, fake_gen)
        self.assertEqual(call_count, 1)
        
    def test_09_cache_hit_no_recall(self):
        call_count = 0
        def fake_gen(q, n):
            nonlocal call_count
            call_count += 1
            return [{"text": "B", "focus": "p"}]
        res1 = generate_query_variants("A", self.config, fake_gen)
        res2 = generate_query_variants("A", self.config, fake_gen)
        self.assertEqual(call_count, 1)
        self.assertTrue(res2.get("cache_hit", False))
        
    def test_10_api_error_status(self):
        def fake_gen(q, n):
            raise ConnectionError("Network down")
        res = generate_query_variants("A", self.config, fake_gen)
        self.assertEqual(res["status"], "query_generation_unavailable")
        
    def test_11_no_network_call(self):
        # This test ensures we pass when custom generator is provided
        res = generate_query_variants("Test", self.config, lambda q, n: [{"text": "V", "focus": "p"}])
        self.assertEqual(res["status"], "ready")

if __name__ == "__main__":
    unittest.main()
