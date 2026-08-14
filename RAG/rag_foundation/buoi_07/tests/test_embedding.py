import unittest
import math
from rag import validate_embeddings

class TestEmbedding(unittest.TestCase):
    def test_case_15_wrong_number_of_vectors_fails(self):
        with self.assertRaises(ValueError):
            validate_embeddings([[1.0]*128], expected_len=2, expected_dim=128)

    def test_case_16_empty_vector_fails(self):
        with self.assertRaises(ValueError):
            validate_embeddings([[]], expected_len=1, expected_dim=128)

    def test_case_17_wrong_dimension_fails(self):
        with self.assertRaises(ValueError):
            validate_embeddings([[1.0]*127], expected_len=1, expected_dim=128)

    def test_case_18_nan_inf_fails(self):
        with self.assertRaises(ValueError):
            validate_embeddings([[float('nan')] + [1.0]*127], expected_len=1, expected_dim=128)
        with self.assertRaises(ValueError):
            validate_embeddings([[float('inf')] + [1.0]*127], expected_len=1, expected_dim=128)

    def test_case_39_boolean_and_zero_vector_fails(self):
        with self.assertRaises(ValueError):
            validate_embeddings([[True] + [1.0]*127], expected_len=1, expected_dim=128)
        with self.assertRaises(ValueError):
            validate_embeddings([[0.0]*128], expected_len=1, expected_dim=128)
