import unittest
from evaluate import compute_recall_at_k, compute_mrr_at_k, compute_ndcg_at_k

class TestEvaluate(unittest.TestCase):
    def test_recall_at_k(self):
        rel = ["A", "B", "C"]
        ret = ["X", "A", "Y", "C", "Z"]
        # k=3 -> ["X", "A", "Y"] -> hits = "A" (1) -> recall = 1/3
        self.assertAlmostEqual(compute_recall_at_k(ret, rel, 3), 1/3)
        # k=5 -> ["X", "A", "Y", "C", "Z"] -> hits = "A", "C" (2) -> recall = 2/3
        self.assertAlmostEqual(compute_recall_at_k(ret, rel, 5), 2/3)
        
    def test_mrr_at_k(self):
        rel = ["A", "B"]
        ret = ["X", "Y", "A", "Z"]
        # k=2 -> ["X", "Y"] -> no hit -> MRR = 0
        self.assertAlmostEqual(compute_mrr_at_k(ret, rel, 2), 0.0)
        # k=4 -> hit at rank 3 -> MRR = 1/3
        self.assertAlmostEqual(compute_mrr_at_k(ret, rel, 4), 1/3)
        
    def test_ndcg_at_k(self):
        import math
        rel = ["A", "B"]
        ret = ["A", "X", "B", "Y"]
        
        # k=3 -> ["A", "X", "B"] -> hits at rank 1, 3
        dcg = 1.0 / math.log2(1+1) + 1.0 / math.log2(3+1) # 1.0 + 0.5 = 1.5
        idcg = 1.0 / math.log2(1+1) + 1.0 / math.log2(2+1) # 1.0 + 0.6309 = 1.6309
        expected = dcg / idcg
        self.assertAlmostEqual(compute_ndcg_at_k(ret, rel, 3), expected)

if __name__ == '__main__':
    unittest.main()
