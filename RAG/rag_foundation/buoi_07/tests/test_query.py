import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from rag import run_index, query_rag, get_collection_name

def mock_query_embed_fn(question, api_key, model, dim):
    return [[0.1] * dim]

class TestQuery(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_col = MagicMock()
        self.mock_client.get_collection.return_value = self.mock_col
        self.mock_client.create_collection.return_value = self.mock_col
        self.mock_col.count.return_value = 10
        self.mock_col.metadata = {"strategy": "semantic", "embedding_model": "mock-emb-model", "embedding_dim": 128, "distance_metric": "cosine"}
        
        mock_c_obj = MagicMock()
        mock_c_obj.name = get_collection_name("semantic", 128, "mock-emb-model")
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
            "max_dist": 0.45,
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
        
    def mock_gen_fn_success(self, prompt):
        return "Trả lời câu hỏi [E1] và [E2]."
        
    def mock_gen_fn_empty(self, prompt):
        return ""
        
    def mock_gen_fn_error(self, prompt):
        raise ValueError("Mock Gen Error")
        
    def mock_gen_fn_invalid_label(self, prompt):
        return "Trả lời [E1] và label giả [E99]."

    def test_case_24_invalid_question_topk(self):
        with self.assertRaises(ValueError):
            query_rag("", 5, "semantic", storage_dir=self.storage_dir, _config=self.config)
        with self.assertRaises(ValueError):
            query_rag("q", 0, "semantic", storage_dir=self.storage_dir, _config=self.config)
        with self.assertRaises(ValueError):
            query_rag("q", 21, "semantic", storage_dir=self.storage_dir, _config=self.config)

    def test_case_25_invalid_question_topk(self):
        with self.assertRaises(ValueError):
            query_rag("", 5, "semantic", storage_dir=self.storage_dir, _config=self.config)
        with self.assertRaises(ValueError):
            query_rag("q", 0, "semantic", storage_dir=self.storage_dir, _config=self.config)
        with self.assertRaises(ValueError):
            query_rag("q", 21, "semantic", storage_dir=self.storage_dir, _config=self.config)

    
    def test_case_26_empty_collection(self):
        self.mock_col.count.return_value = 0
        with self.assertRaises(ValueError):
            query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config)

    
    def test_case_14_mismatch_metadata(self):
        self.mock_col.metadata = {"strategy": "wrong"}
        with self.assertRaises(ValueError):
            query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config)

    
    def test_case_21_retrieval(self):
        self._setup_query_mock(2, [0.1, 0.2])
        res = query_rag("q", 2, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(len(res["evidence"]), 2)

    def test_case_22_retrieval(self):
        self._setup_query_mock(2, [0.1, 0.2])
        res = query_rag("q", 2, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(len(res["evidence"]), 2)

    def test_case_23_retrieval(self):
        self._setup_query_mock(2, [0.1, 0.2])
        res = query_rag("q", 2, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(len(res["evidence"]), 2)

    
    def test_case_27_best_evidence_exceeds_threshold(self):
        self._setup_query_mock(1, [1.0])
        self.config["max_dist"] = -0.1
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["citations"], [])

    
    def test_case_36_generation_error_and_empty(self):
        self._setup_query_mock(1)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_error)
        self.assertEqual(res["status"], "retrieval_only")
        
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_empty)
        self.assertEqual(res["status"], "retrieval_only")

    def test_case_46_generation_error_and_empty(self):
        self._setup_query_mock(1)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_error)
        self.assertEqual(res["status"], "retrieval_only")
        
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_empty)
        self.assertEqual(res["status"], "retrieval_only")

    
    def test_case_43_mixed_threshold(self):
        self._setup_query_mock(2, [0.1, 0.9])
        self.config["max_dist"] = 0.5
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertTrue(res["evidence"][0]["accepted"])
        self.assertFalse(res["evidence"][1]["accepted"])
        
    
    def test_case_32_citations(self):
        self._setup_query_mock(3)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_invalid_label)
        
        self.assertEqual(res["status"], "answered")
        self.assertTrue(len(res["citations"]) > 0)
        self.assertTrue("[Nguồn:" in res["answer"])
        self.assertFalse("[E99]" in res["answer"])
        self.assertTrue(any("E99" in w for w in res["warnings"]))

    def test_case_33_citations(self):
        self._setup_query_mock(3)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_invalid_label)
        
        self.assertEqual(res["status"], "answered")
        self.assertTrue(len(res["citations"]) > 0)
        self.assertTrue("[Nguồn:" in res["answer"])
        self.assertFalse("[E99]" in res["answer"])
        self.assertTrue(any("E99" in w for w in res["warnings"]))

    def test_case_34_citations(self):
        self._setup_query_mock(3)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_invalid_label)
        
        self.assertEqual(res["status"], "answered")
        self.assertTrue(len(res["citations"]) > 0)
        self.assertTrue("[Nguồn:" in res["answer"])
        self.assertFalse("[E99]" in res["answer"])
        self.assertTrue(any("E99" in w for w in res["warnings"]))

    def test_case_35_citations(self):
        self._setup_query_mock(3)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_invalid_label)
        
        self.assertEqual(res["status"], "answered")
        self.assertTrue(len(res["citations"]) > 0)
        self.assertTrue("[Nguồn:" in res["answer"])
        self.assertFalse("[E99]" in res["answer"])
        self.assertTrue(any("E99" in w for w in res["warnings"]))

    def test_case_45_citations(self):
        self._setup_query_mock(3)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_invalid_label)
        
        self.assertEqual(res["status"], "answered")
        self.assertTrue(len(res["citations"]) > 0)
        self.assertTrue("[Nguồn:" in res["answer"])
        self.assertFalse("[E99]" in res["answer"])
        self.assertTrue(any("E99" in w for w in res["warnings"]))

    
    def test_case_37_result_structure(self):
        self._setup_query_mock(1)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        for key in ["status", "answer", "evidence", "citations", "warnings", "collection", "strategy", "top_k"]:
            self.assertIn(key, res)

    
    def test_case_28_prompt_contains(self):
        self._setup_query_mock(2)
        self.config["max_dist"] = 10.0
        
        def mock_gen_check_prompt(prompt):
            self.assertTrue("q_text" in prompt)
            self.assertTrue("t1" in prompt)
            self.assertTrue("t2" in prompt)
            self.assertTrue("dữ liệu không đáng tin cậy" in prompt.lower())
            return "[E1]"
            
        query_rag("q_text", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_check_prompt)

    def test_case_29_prompt_contains(self):
        self._setup_query_mock(2)
        self.config["max_dist"] = 10.0
        
        def mock_gen_check_prompt(prompt):
            self.assertTrue("q_text" in prompt)
            self.assertTrue("t1" in prompt)
            self.assertTrue("t2" in prompt)
            self.assertTrue("dữ liệu không đáng tin cậy" in prompt.lower())
            return "[E1]"
            
        query_rag("q_text", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_check_prompt)

    def test_case_30_prompt_contains(self):
        self._setup_query_mock(2)
        self.config["max_dist"] = 10.0
        
        def mock_gen_check_prompt(prompt):
            self.assertTrue("q_text" in prompt)
            self.assertTrue("t1" in prompt)
            self.assertTrue("t2" in prompt)
            self.assertTrue("dữ liệu không đáng tin cậy" in prompt.lower())
            return "[E1]"
            
        query_rag("q_text", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_check_prompt)

    def test_case_31_prompt_contains(self):
        self._setup_query_mock(2)
        self.config["max_dist"] = 10.0
        
        def mock_gen_check_prompt(prompt):
            self.assertTrue("q_text" in prompt)
            self.assertTrue("t1" in prompt)
            self.assertTrue("t2" in prompt)
            self.assertTrue("dữ liệu không đáng tin cậy" in prompt.lower())
            return "[E1]"
            
        query_rag("q_text", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_check_prompt)

    def test_case_44_prompt_contains(self):
        self._setup_query_mock(2)
        self.config["max_dist"] = 10.0
        
        def mock_gen_check_prompt(prompt):
            self.assertTrue("q_text" in prompt)
            self.assertTrue("t1" in prompt)
            self.assertTrue("t2" in prompt)
            self.assertTrue("dữ liệu không đáng tin cậy" in prompt.lower())
            return "[E1]"
            
        query_rag("q_text", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_check_prompt)

    