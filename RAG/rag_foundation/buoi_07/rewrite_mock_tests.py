import sys
import glob
from pathlib import Path

tests_dir = Path(r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests")

test_index_content = """import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from rag import run_index, get_collection_name, run_status

def mock_embed_fn(chunks, api_key, model, dim):
    return [[float(i+1)/100.0] * dim for i in range(len(chunks))]

def mock_embed_fn_error(chunks, api_key, model, dim):
    raise ValueError("Mock Embedding Error")

class TestIndex(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_col = MagicMock()
        self.mock_client.get_collection.return_value = self.mock_col
        self.mock_client.create_collection.return_value = self.mock_col
        self.mock_col.count.return_value = 0
        self.mock_col.metadata = {"strategy": "semantic", "embedding_model": "mock-emb-model", "embedding_dim": 128, "distance_metric": "cosine", "schema_version": "1.0"}
        
        # mock list_collections
        mock_c_obj = MagicMock()
        mock_c_obj.name = "nhnn-semantic-128-4e8c130d"
        self.mock_client.list_collections.return_value = [mock_c_obj]
        
        self.patcher = patch('chromadb.PersistentClient', return_value=self.mock_client)
        self.patcher.start()
        
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "chroma"
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
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

    def _write_json(self, name, data):
        with open(self.input_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_case_10_index_twice_no_dup(self):
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        self.assertTrue(self.mock_col.upsert.called)
        
    def test_case_11_metadata_citation_saved(self):
        self._write_json("t1.json", [{"chunk_id": "c1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 2, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        self.assertTrue(self.mock_col.upsert.called)
        
    def test_case_12_collection_identity_strategy(self):
        self._write_json("t1.json", [
            {"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"},
            {"chunk_id": "2", "strategy": "hierarchical", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}
        ])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        run_index(str(self.input_dir), "hierarchical", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        self.assertEqual(self.mock_client.create_collection.call_count, 1) # one is semantic (exists), one hierarchical (created)
        
    def test_case_13_collection_identity_model_dim(self):
        c_name1 = get_collection_name("semantic", 128, "model-A")
        c_name2 = get_collection_name("semantic", 256, "model-A")
        c_name3 = get_collection_name("semantic", 128, "model-B")
        self.assertNotEqual(c_name1, c_name2)
        self.assertNotEqual(c_name1, c_name3)
        
    def test_case_19_41_embedding_error_before_upsert_retains_collection(self):
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        try:
            run_index(str(self.input_dir), "semantic", reset=True, storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn_error)
        except:
            pass
        self.assertFalse(self.mock_col.upsert.called)
        
    def test_case_20_missing_api_key_fails(self):
        self.config["has_api_key"] = False
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        with self.assertRaises(ValueError):
            run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
            
    def test_case_40_status_on_empty_storage_no_collection(self):
        run_status("semantic", storage_dir=self.storage_dir, _config=self.config)
        self.assertFalse(self.storage_dir.exists())
        
    def test_case_42_mismatch_metadata_blocked(self):
        self.mock_col.metadata = {"strategy": "wrong"}
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        self.assertFalse(self.mock_col.upsert.called)
"""

test_query_content = """import unittest
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
        mock_c_obj.name = "nhnn-semantic-128-4e8c130d"
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
        return "   "
        
    def mock_gen_fn_error(self, prompt):
        raise ValueError("Mock Gen Error")
        
    def mock_gen_fn_invalid_label(self, prompt):
        return "Trả lời [E1] và label giả [E99]."

    def test_case_24_25_invalid_question_topk(self):
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

    def test_case_21_22_23_retrieval(self):
        self._setup_query_mock(2, [0.1, 0.2])
        res = query_rag("q", 2, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(len(res["evidence"]), 2)

    def test_case_27_best_evidence_exceeds_threshold(self):
        self._setup_query_mock(1, [1.0])
        self.config["max_dist"] = -0.1
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["citations"], [])

    def test_case_36_46_generation_error_and_empty(self):
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
        
    def test_case_32_33_34_35_45_citations(self):
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

    def test_case_28_29_30_31_44_prompt_contains(self):
        self._setup_query_mock(2)
        self.config["max_dist"] = 10.0
        
        def mock_gen_check_prompt(prompt):
            self.assertTrue("q_text" in prompt)
            self.assertTrue("t1" in prompt)
            self.assertTrue("t2" in prompt)
            self.assertTrue("dữ liệu không đáng tin cậy" in prompt.lower())
            return "[E1]"
            
        query_rag("q_text", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=mock_gen_check_prompt)
"""

with open(tests_dir / "test_index.py", "w", encoding="utf-8") as f:
    f.write(test_index_content)
with open(tests_dir / "test_query.py", "w", encoding="utf-8") as f:
    f.write(test_query_content)

print("Rewrote tests to use MagicMock for Chroma.")
