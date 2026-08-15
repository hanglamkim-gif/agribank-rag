import unittest
from advanced_rag import tokenize_vi_legal, search_bm25, build_bm25_index

class TestBM25(unittest.TestCase):
    
    def test_tokenizer_keeps_vietnamese(self):
        # 1. Tokenizer giữ dấu tiếng Việt.
        text = "cơ cấu lại thời hạn trả nợ"
        tokens = tokenize_vi_legal(text)
        self.assertEqual(tokens, ["cơ", "cấu", "lại", "thời", "hạn", "trả", "nợ"])

    def test_tokenizer_keeps_legal_numbers(self):
        # 2. Tokenizer giữ số Điều/Khoản.
        text = "Điều 7, Khoản 2"
        tokens = tokenize_vi_legal(text)
        self.assertEqual(tokens, ["điều", "7", "khoản", "2"])
        
    def test_bm25_search_exact_legal_term(self):
        # 4. Exact legal term được sắp trên đoạn không chứa từ khóa.
        chunks = [
            {"chunk_id": "C1", "text": "Quy định về bảo mật thông tin nội bộ của công ty."},
            {"chunk_id": "C2", "text": "Điều 7 Khoản 2 quy định về thời gian làm việc."},
            {"chunk_id": "C3", "text": "Nội quy công ty quy định rõ ràng."}
        ]
        results = search_bm25("Điều 7 Khoản 2", chunks, candidate_k=10)
        self.assertEqual(results[0]["chunk_id"], "C2")
        self.assertTrue(results[0]["bm25_score"] > results[1]["bm25_score"])

    def test_bm25_candidate_k_larger_than_corpus(self):
        # 5. candidate_k lớn hơn corpus vẫn chạy.
        chunks = [
            {"chunk_id": "C1", "text": "A"}
        ]
        results = search_bm25("A", chunks, candidate_k=100)
        self.assertEqual(len(results), 1)

    def test_bm25_empty_question_fails(self):
        # 6. Empty question fail.
        chunks = [{"chunk_id": "C1", "text": "A"}]
        with self.assertRaises(ValueError):
            search_bm25("", chunks, 10)
        with self.assertRaises(ValueError):
            search_bm25("   ", chunks, 10)
        with self.assertRaises(ValueError):
            search_bm25("...", chunks, 10) # No valid tokens

    def test_bm25_tie_break_deterministic(self):
        # 7. Tie-break deterministic.
        chunks = [
            {"chunk_id": "C3", "text": "nhân viên"},
            {"chunk_id": "C1", "text": "nhân viên"},
            {"chunk_id": "C2", "text": "nhân viên"}
        ]
        results = search_bm25("nhân viên", chunks, candidate_k=10)
        # Điểm bằng nhau, chunk_id nhỏ hơn đứng trước (C1, C2, C3)
        self.assertEqual([r["chunk_id"] for r in results], ["C1", "C2", "C3"])

    def test_no_external_calls(self):
        # 8. Không gọi Gemini/Chroma/reranker.
        # BM25 is purely local math. If it called APIs, it would fail here or take long.
        chunks = [{"chunk_id": "C1", "text": "hello"}]
        results = search_bm25("hello", chunks, 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "C1")

if __name__ == '__main__':
    unittest.main()
