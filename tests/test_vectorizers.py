import os
import unittest
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
ENV_FILE = f"{BASE_DIR}/.env"

load_dotenv(ENV_FILE)

sys.path.append(BASE_DIR)

from core.vectorizers import (
    SIFTextVectorizer,
    SentBERTVectorizer,
    CPCVectorizer,
    EmbeddingMatrix,
    BagOfVectorsEncoder
)


class TestSIFTextVectorizer(unittest.TestCase):

    """Test functionality of `SIFTextVectorizer` class"""

    def test__can_encode_one(self):
        sent = "This invention is a mouse trap."
        output = SIFTextVectorizer().encode(sent)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 1)
        self.assertGreater(output.shape[0], 0)

    def test__can_encode_multiple(self):
        sents = [
            "This invention is a mouse trap.",
            "This invention presents a bird cage.",
        ]
        output = SIFTextVectorizer().encode_many(sents)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 2)
        self.assertGreater(output.shape[1], 0)


class TestSentBERTVectorizer(unittest.TestCase):

    """Test functionality of `SentBERTVectorizer` class"""

    def test__can_encode_one(self):
        sent = "This invention is a mouse trap."
        output = SentBERTVectorizer().encode(sent)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 1)
        self.assertGreater(output.shape[0], 0)

    def test__can_encode_multiple(self):
        sents = [
            "This invention is a mouse trap.",
            "This invention presents a bird cage.",
        ]
        output = SentBERTVectorizer().encode_many(sents)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 2)
        self.assertGreater(output.shape[1], 0)


class TestCPCVectorizer(unittest.TestCase):

    """Test functionality of `CPCVectorizer` class"""

    def test__can_encode_one(self):
        cpc = "H04W52/02"
        output = CPCVectorizer().encode(cpc)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 1)
        self.assertGreater(output.shape[0], 0)

    def test__can_encode_multiple(self):
        cpcs = ["H04W52/02", "H04W72/00"]
        output = CPCVectorizer().encode_many(cpcs)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 2)
        self.assertGreater(output.shape[1], 0)


class TestEmbeddingMatrixClass(unittest.TestCase):

    """Test functionality of `EmbeddingMatrix` class"""

    def setUp(self):
        file = f'{TEST_DIR}/test_embs.tsv'
        self.E = EmbeddingMatrix.from_tsv(file)

    def test__can_return_embedding_dimensions(self):
        n_dim = self.E.dims
        self.assertEqual(2, n_dim)

    def test__can_determine_if_a_label_has_embedding(self):
        self.assertTrue("base" in self.E)
        self.assertFalse("django" in self.E)

    def test__can_return_word_vector(self):
        word = 'base'
        expected = [1.0, 0.0]
        actual = list(self.E[word])
        self.assertEqual(expected, actual)

    def test__can_return_similar_items(self):
        item = 'station'
        similars = self.E.similar_to_item(item)
        most_similar = similars[1]  # [0] is the item itself
        self.assertEqual('stations', most_similar)


class TestBagOfVectorsEncoder(unittest.TestCase):

    def setUp(self):
        emb_matrix_file = f'{TEST_DIR}/test_embs.tsv'
        emb_matrix = EmbeddingMatrix.from_tsv(emb_matrix_file)
        self.encoder = BagOfVectorsEncoder(emb_matrix)
        
    def test_can_encode_simple_entity_set(self):
        entities = set([ 'base', 'station' ])
        base_vec = tuple([1.0, 0.0])
        station_vec = tuple([0.1, 2.0])
        expected_bov = set([base_vec, station_vec])
        bov = self.encoder.encode(entities)
        self.assertEqual(expected_bov, bov)


if __name__ == "__main__":
    unittest.main()
