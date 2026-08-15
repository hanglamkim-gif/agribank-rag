import unittest
import os
import shutil
import json
from pathlib import Path

from hierarchical_rag import resolve_hierarchy, build_parents, load_buoi09_config, _make_stable_id

class TestHierarchy(unittest.TestCase):
    def setUp(self):
        self.config = {
            "PARENT_MAX_CHARS": 1000,
            "PARENT_SCORE_CHILD_LIMIT": 3,
            "PARENT_RRF_K": 60,
            "PARENT_CANDIDATES": 10,
            "FINAL_PARENT_TOP_K": 3,
            "TOTAL_CONTEXT_MAX_CHARS": 2000
        }
        
    def test_01_metadata_precedence(self):
        # Rule 1: Metadata wins over heading
        chunks = [{
            "chunk_id": "c1:1", "strategy": "hierarchical", "source": "f1.pdf",
            "page_start": 1, "page_end": 1,
            "text": "Điều 2. Something",
            "metadata": {"structure": {"article": "Điều 10"}}
        }]
        res = resolve_hierarchy(chunks)
        self.assertEqual(res[0]["resolution_method"], "metadata")
        self.assertEqual(res[0]["structural_path"]["article"], "Điều 10")
        self.assertTrue(res[0]["ambiguous"]) # because Dieu 2 != Dieu 10
        self.assertEqual(len(res[0]["warnings"]), 1)
        
    def test_02_heading_inferred(self):
        # Rule 2: Heading at start of string
        chunks = [{
            "chunk_id": "c2:1", "strategy": "hierarchical", "source": "f2.pdf",
            "page_start": 1, "page_end": 1,
            "text": "Điều 5. Test"
        }]
        res = resolve_hierarchy(chunks)
        self.assertEqual(res[0]["resolution_method"], "heading_inferred")
        self.assertEqual(res[0]["structural_path"]["article"], "Điều 5")
        
    def test_03_carry_forward(self):
        chunks = [
            {
                "chunk_id": "c3:1", "strategy": "h", "source": "f3",
                "page_start": 1, "page_end": 1, "text": "Điều 1. A"
            },
            {
                "chunk_id": "c3:2", "strategy": "h", "source": "f3",
                "page_start": 1, "page_end": 1, "text": "Khoản 2. B"
            }
        ]
        res = resolve_hierarchy(chunks)
        self.assertEqual(res[1]["resolution_method"], "carried_forward")
        self.assertEqual(res[1]["structural_path"]["article"], "Điều 1")
        
    def test_04_no_carry_cross_source(self):
        chunks = [
            {"chunk_id": "c4:1", "strategy": "h", "source": "fA", "page_start": 1, "page_end": 1, "text": "Điều 1. A"},
            {"chunk_id": "c4:2", "strategy": "h", "source": "fB", "page_start": 1, "page_end": 1, "text": "Khoản 2. B"}
        ]
        res = resolve_hierarchy(chunks)
        self.assertEqual(res[1]["resolution_method"], "document_fallback")
        
    def test_05_inline_not_inferred(self):
        chunks = [{
            "chunk_id": "c5:1", "strategy": "h", "source": "f5",
            "page_start": 1, "page_end": 1,
            "text": "Theo Điều 5 của luật, ta có..."
        }]
        res = resolve_hierarchy(chunks)
        self.assertEqual(res[0]["resolution_method"], "document_fallback")
        self.assertEqual(res[0]["structural_path"]["article"], "Fallback")
        
    def test_06_conflict_warning(self):
        chunks = [{
            "chunk_id": "c6:1", "strategy": "h", "source": "f6",
            "page_start": 1, "page_end": 1,
            "text": "Điều 1. A\nĐiều 2. B"
        }]
        res = resolve_hierarchy(chunks)
        self.assertTrue(res[0]["ambiguous"])
        self.assertEqual(len(res[0]["warnings"]), 1)
        self.assertEqual(res[0]["resolution_method"], "heading_inferred")
        
    def test_07_numeric_chunk_ordering(self):
        chunks = [
            {"chunk_id": "c7:10", "strategy": "h", "source": "f7", "page_start": 1, "page_end": 1, "text": "B"},
            {"chunk_id": "c7:2", "strategy": "h", "source": "f7", "page_start": 1, "page_end": 1, "text": "Điều 1. A"}
        ]
        res = resolve_hierarchy(chunks)
        self.assertEqual(res[0]["child_id"], "c7:2")
        self.assertEqual(res[1]["child_id"], "c7:10")
        self.assertEqual(res[1]["structural_path"]["article"], "Điều 1")
        
    def test_08_stable_parent_id(self):
        res1 = [{"child_id": "c8", "source": "f8", "page_start":1, "page_end":1, "text":"A", "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]}]
        p1 = build_parents(res1, self.config)
        p2 = build_parents(res1, self.config)
        self.assertEqual(p1[0]["parent_id"], p2[0]["parent_id"])
        expected_id = _make_stable_id("f8", "None::Điều 1", 1)
        self.assertEqual(p1[0]["parent_id"], expected_id)
        
    def test_09_parent_split(self):
        res = [
            {"child_id": "1", "source": "f", "page_start":1, "page_end":1, "text": "A" * 600, "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]},
            {"child_id": "2", "source": "f", "page_start":1, "page_end":1, "text": "B" * 600, "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]}
        ]
        parents = build_parents(res, self.config) # limit 1000
        self.assertEqual(len(parents), 2)
        self.assertEqual(parents[0]["window_index"], 1)
        self.assertEqual(parents[1]["window_index"], 2)
        
    def test_10_oversized_child(self):
        res = [{"child_id": "1", "source": "f", "page_start":1, "page_end":1, "text": "A" * 1500, "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]}]
        parents = build_parents(res, self.config)
        self.assertEqual(len(parents), 1)
        self.assertIn("oversized_single_child", parents[0]["warnings"])
        
    def test_11_each_child_one_parent(self):
        res = [
            {"child_id": "1", "source": "f", "page_start":1, "page_end":1, "text": "A" * 600, "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]},
            {"child_id": "2", "source": "f", "page_start":1, "page_end":1, "text": "B" * 600, "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]}
        ]
        parents = build_parents(res, self.config)
        self.assertEqual(res[0]["parent_id"], parents[0]["parent_id"])
        self.assertEqual(res[1]["parent_id"], parents[1]["parent_id"])
        
    def test_12_parent_pages_text(self):
        res = [
            {"child_id": "1", "source": "f", "page_start":1, "page_end":1, "text": "ABC", "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]},
            {"child_id": "2", "source": "f", "page_start":2, "page_end":3, "text": "DEF", "structural_path":{"article":"Điều 1"}, "ambiguous":False, "warnings":[]}
        ]
        parents = build_parents(res, self.config)
        p = parents[0]
        self.assertEqual(p["page_start"], 1)
        self.assertEqual(p["page_end"], 3)
        self.assertEqual(p["text"], "ABC\nDEF")
        self.assertEqual(p["char_count"], 7)
        
    def test_13_atomic_build_manifest(self):
        # We will mock load_chunks to test cmd_build_hierarchy
        import hierarchical_rag
        import datetime
        original_load = hierarchical_rag.load_chunks
        original_config = hierarchical_rag.load_buoi09_config
        
        try:
            hierarchical_rag.load_buoi09_config = lambda: self.config
            hierarchical_rag.load_chunks = lambda x: [
                {"chunk_id": "c:1", "strategy": "hierarchical", "source": "src", "page_start": 1, "page_end": 1, "text": "Điều 1"}
            ]
            hierarchical_rag.cmd_build_hierarchy()
            
            # Check files exist
            base = Path(hierarchical_rag.__file__).resolve().parent / "storage" / "hierarchy"
            self.assertTrue((base / "manifest.json").exists())
            self.assertTrue((base / "parents.json").exists())
            self.assertTrue((base / "children.json").exists())
            
            with open(base / "manifest.json", "r", encoding="utf-8") as f:
                m = json.load(f)
                self.assertEqual(m["child_count"], 1)
                self.assertEqual(m["parent_count"], 1)
                
        finally:
            hierarchical_rag.load_chunks = original_load
            hierarchical_rag.load_buoi09_config = original_config

    def test_14_status_read_only(self):
        import hierarchical_rag
        import io
        import sys
        
        base = Path(hierarchical_rag.__file__).resolve().parent / "storage" / "hierarchy"
        if base.exists():
            shutil.rmtree(base)
            
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        hierarchical_rag.cmd_hierarchy_status()
        sys.stdout = sys.__stdout__
        
        self.assertIn("Hierarchy Ready: False", capturedOutput.getvalue())
        self.assertFalse(base.exists()) # status does not create dir
        
if __name__ == "__main__":
    unittest.main()
