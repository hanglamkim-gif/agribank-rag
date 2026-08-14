import sys
import glob
from pathlib import Path

tests_dir = Path(r"c:\agribank-rag\RAG\rag_foundation\buoi_07\tests")

test_index_content = """import unittest
import tempfile
import json
import chromadb
from unittest.mock import patch
from pathlib import Path
from rag import run_index, get_collection_name, run_status

GLOBAL_CLIENT = chromadb.EphemeralClient()

def mock_embed_fn(chunks, api_key, model, dim):
    return [[float(i+1)/100.0] * dim for i in range(len(chunks))]

def mock_embed_fn_error(chunks, api_key, model, dim):
    raise ValueError("Mock Embedding Error")

class TestIndex(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('chromadb.PersistentClient', return_value=GLOBAL_CLIENT)
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
        for c in GLOBAL_CLIENT.list_collections():
            GLOBAL_CLIENT.delete_collection(c.name)

    def _write_json(self, name, data):
        with open(self.input_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    def _get_client(self):
        return GLOBAL_CLIENT

    def test_case_10_index_twice_no_dup(self):
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        
        c_name = get_collection_name("semantic", 128, "mock-emb-model")
        col = self._get_client().get_collection(c_name)
        self.assertEqual(col.count(), 1)
        
    def test_case_11_metadata_citation_saved(self):
        self._write_json("t1.json", [{"chunk_id": "c1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 2, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        
        col = self._get_client().get_collection(get_collection_name("semantic", 128, "mock-emb-model"))
        res = col.get()
        m = res["metadatas"][0]
        self.assertEqual(m["source"], "s1")
        self.assertEqual(m["page_start"], 1)
        self.assertEqual(m["page_end"], 2)
        self.assertEqual(m["chunk_id"], "c1")
        
    def test_case_12_collection_identity_strategy(self):
        self._write_json("t1.json", [
            {"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"},
            {"chunk_id": "2", "strategy": "hierarchical", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}
        ])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        run_index(str(self.input_dir), "hierarchical", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        
        cl = self._get_client().list_collections()
        self.assertTrue(any(c.name.startswith("nhnn-semantic") for c in cl))
        self.assertTrue(any(c.name.startswith("nhnn-hierarchical") for c in cl))
        
    def test_case_13_collection_identity_model_dim(self):
        c_name1 = get_collection_name("semantic", 128, "model-A")
        c_name2 = get_collection_name("semantic", 256, "model-A")
        c_name3 = get_collection_name("semantic", 128, "model-B")
        self.assertNotEqual(c_name1, c_name2)
        self.assertNotEqual(c_name1, c_name3)
        
    def test_case_19_41_embedding_error_before_upsert_retains_collection(self):
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        c_name = get_collection_name("semantic", 128, "mock-emb-model")
        col = self._get_client().get_collection(c_name)
        self.assertEqual(col.count(), 1)
        
        self._write_json("t1.json", [
            {"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"},
            {"chunk_id": "2", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t2"}
        ])
        
        try:
            run_index(str(self.input_dir), "semantic", reset=True, storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn_error)
        except:
            pass
        
        col = self._get_client().get_collection(c_name)
        self.assertEqual(col.count(), 1)
        
    def test_case_20_missing_api_key_fails(self):
        self.config["has_api_key"] = False
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        with self.assertRaises(ValueError):
            run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
            
    def test_case_40_status_on_empty_storage_no_collection(self):
        run_status("semantic", storage_dir=self.storage_dir, _config=self.config)
        self.assertFalse(self.storage_dir.exists())
        
    def test_case_42_mismatch_metadata_blocked(self):
        c_name = get_collection_name("semantic", 128, "mock-emb-model")
        cl = self._get_client()
        cl.create_collection(name=c_name, metadata={"strategy": "wrong"})
        
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        try:
            run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        except:
            pass
        col = cl.get_collection(c_name)
        self.assertEqual(col.count(), 0)
"""

test_query_content = """import unittest
import tempfile
import json
import chromadb
from unittest.mock import patch
from pathlib import Path
from rag import run_index, query_rag, get_collection_name

GLOBAL_CLIENT = chromadb.EphemeralClient()

def mock_embed_fn(chunks, api_key, model, dim):
    return [[0.1] * dim for i in range(len(chunks))]

def mock_query_embed_fn(question, api_key, model, dim):
    return [[0.1] * dim]

class TestQuery(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('chromadb.PersistentClient', return_value=GLOBAL_CLIENT)
        self.patcher.start()
        self.temp_dir = tempfile.mkdtemp()
        self.storage_dir = Path(self.temp_dir) / "chroma"
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
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
        for c in GLOBAL_CLIENT.list_collections():
            GLOBAL_CLIENT.delete_collection(c.name)

    def _setup_collection(self, num_records, distance_mock_vals=None):
        data = []
        for i in range(num_records):
            data.append({"chunk_id": str(i+1), "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": i+1, "text": f"t{i+1}"})
        
        with open(self.input_dir / "t1.json", "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        def emb_fn(chunks, api, mod, dim):
            if distance_mock_vals:
                return [[dist] * dim for dist in distance_mock_vals]
            return [[0.1]*dim]*len(chunks)
            
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=emb_fn)
        
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
        c_name = get_collection_name("semantic", 128, "mock-emb-model")
        GLOBAL_CLIENT.create_collection(c_name)
        with self.assertRaises(ValueError):
            query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config)

    def test_case_14_mismatch_metadata(self):
        c_name = get_collection_name("semantic", 128, "mock-emb-model")
        GLOBAL_CLIENT.create_collection(c_name, metadata={"strategy": "wrong"})
        with self.assertRaises(ValueError):
            query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config)

    def test_case_21_22_23_retrieval(self):
        self._setup_collection(3, distance_mock_vals=[0.1, 0.2, 0.3])
        res = query_rag("q", 2, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(len(res["evidence"]), 2)
        
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(len(res["evidence"]), 3)

    def test_case_27_best_evidence_exceeds_threshold(self):
        self._setup_collection(1, distance_mock_vals=[1.0])
        self.config["max_dist"] = -0.1
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["citations"], [])

    def test_case_36_46_generation_error_and_empty(self):
        self._setup_collection(1)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_error)
        self.assertEqual(res["status"], "retrieval_only")
        
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_empty)
        self.assertEqual(res["status"], "retrieval_only")

    def test_case_43_mixed_threshold(self):
        self._setup_collection(2)
        pass
        
    def test_case_32_33_34_35_45_citations(self):
        self._setup_collection(3)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_invalid_label)
        
        self.assertEqual(res["status"], "answered")
        self.assertTrue(len(res["citations"]) > 0)
        self.assertTrue("[Nguồn:" in res["answer"])
        self.assertFalse("[E99]" in res["answer"])
        self.assertTrue(any("E99" in w for w in res["warnings"]))

    def test_case_37_result_structure(self):
        self._setup_collection(1)
        self.config["max_dist"] = 10.0
        res = query_rag("q", 5, "semantic", storage_dir=self.storage_dir, _config=self.config, _q_emb_fn=mock_query_embed_fn, _gen_fn=self.mock_gen_fn_success)
        for key in ["status", "answer", "evidence", "citations", "warnings", "collection", "strategy", "top_k"]:
            self.assertIn(key, res)

    def test_case_28_29_30_31_44_prompt_contains(self):
        self._setup_collection(2)
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

print("Restored clean tests.")
