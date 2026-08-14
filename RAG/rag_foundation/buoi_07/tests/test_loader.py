import unittest
import json
import tempfile
from pathlib import Path
from rag import load_chunks, validate_chunk

class TestLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, name, data):
        with open(self.dir_path / name, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_case_01_loader_reads_json_list(self):
        self._write_json("test1.json", [{"chunk_id": "1", "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"}])
        chunks, stats = load_chunks(self.dir_path, "semantic")
        self.assertEqual(len(chunks), 1)

    def test_case_02_loader_reads_object_with_chunks_field(self):
        self._write_json("test2.json", {"chunks": [{"chunk_id": "2", "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"}]})
        chunks, stats = load_chunks(self.dir_path, "semantic")
        self.assertEqual(len(chunks), 1)

    def test_case_03_only_selected_strategy_loaded(self):
        self._write_json("test3.json", [
            {"chunk_id": "1", "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"},
            {"chunk_id": "2", "strategy": "hierarchical", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"}
        ])
        chunks, stats = load_chunks(self.dir_path, "semantic")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "1")

    def test_case_04_missing_required_field_fails(self):
        self._write_json("test4.json", [{"chunk_id": "1", "strategy": "semantic", "source": "src1", "page_start": 1, "text": "abc"}]) # missing page_end
        with self.assertRaises(ValueError):
            load_chunks(self.dir_path, "semantic")

    def test_case_05_wrong_field_type_fails(self):
        self._write_json("test5.json", [{"chunk_id": 123, "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"}]) # id is int
        with self.assertRaises(ValueError):
            load_chunks(self.dir_path, "semantic")

    def test_case_06_boolean_page_fails(self):
        self._write_json("test6.json", [{"chunk_id": "1", "strategy": "semantic", "source": "src1", "page_start": True, "page_end": 1, "text": "abc"}])
        with self.assertRaises(ValueError):
            load_chunks(self.dir_path, "semantic")

    def test_case_07_page_start_gt_page_end_fails(self):
        self._write_json("test7.json", [{"chunk_id": "1", "strategy": "semantic", "source": "src1", "page_start": 2, "page_end": 1, "text": "abc"}])
        with self.assertRaises(ValueError):
            load_chunks(self.dir_path, "semantic")

    def test_case_08_empty_text_skipped_stats(self):
        self._write_json("test8.json", [
            {"chunk_id": "1", "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "   "},
            {"chunk_id": "2", "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"}
        ])
        chunks, stats = load_chunks(self.dir_path, "semantic")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stats["empty_text_skipped"], 1)

    def test_case_09_duplicate_chunk_id_fails(self):
        self._write_json("test9.json", [
            {"chunk_id": "1", "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"},
            {"chunk_id": "1", "strategy": "semantic", "source": "src2", "page_start": 1, "page_end": 1, "text": "def"}
        ])
        with self.assertRaises(ValueError):
            load_chunks(self.dir_path, "semantic")

    def test_case_38_loader_blocks_non_object(self):
        self._write_json("test38.json", ["not_an_object", {"chunk_id": "2", "strategy": "semantic", "source": "src1", "page_start": 1, "page_end": 1, "text": "abc"}])
        with self.assertRaises(ValueError):
            load_chunks(self.dir_path, "semantic")
