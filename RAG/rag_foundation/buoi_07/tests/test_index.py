import unittest
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
        mock_c_obj.name = get_collection_name("semantic", 128, "mock-emb-model")
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
        
    
    def test_case_19_embedding_error_before_upsert_retains_collection(self):
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        try:
            run_index(str(self.input_dir), "semantic", reset=True, storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn_error)
        except:
            pass
        self.assertFalse(self.mock_col.upsert.called)
        
    def test_case_41_embedding_error_before_upsert_retains_collection(self):
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        try:
            run_index(str(self.input_dir), "semantic", reset=True, storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn_error)
        except:
            pass
        self.assertFalse(self.mock_col.upsert.called)
        
    
    def test_case_20_missing_api_key_fails(self):
        self.config["has_api_key"] = False
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        self.assertFalse(self.mock_col.upsert.called)
            
    
    def test_case_40_status_on_empty_storage_no_collection(self):
        run_status("semantic", storage_dir=self.storage_dir, _config=self.config)
        self.assertFalse(self.storage_dir.exists())
        
    
    def test_case_42_mismatch_metadata_blocked(self):
        self.mock_col.metadata = {"strategy": "wrong"}
        self._write_json("t1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "s1", "page_start": 1, "page_end": 1, "text": "t"}])
        run_index(str(self.input_dir), "semantic", storage_dir=self.storage_dir, _config=self.config, _emb_fn=mock_embed_fn)
        self.assertFalse(self.mock_col.upsert.called)

    