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
    CPCVectorizer
)

class TestSIFTextVectorizer(unittest.TestCase):

    def test__can_encode_one(self):
        sent = "This invention is a mouse trap."
        output = SIFTextVectorizer().embed(sent)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 1)
        self.assertGreater(output.shape[0], 0)

    def test__can_encode_multiple(self):
        sents = [
            "This invention is a mouse trap.",
            "This invention presents a bird cage."
        ]
        output = SIFTextVectorizer().encode_many(sents)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 2)
        self.assertGreater(output.shape[1], 0)


class TestSentBERTVectorizer(unittest.TestCase):

    def test__can_encode_one(self):
        sent = "This invention is a mouse trap."
        output = SentBERTVectorizer().embed(sent)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 1)
        self.assertGreater(output.shape[0], 0)

    def test__can_encode_multiple(self):
        sents = [
            "This invention is a mouse trap.",
            "This invention presents a bird cage."
        ]
        output = SentBERTVectorizer().encode_many(sents)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 2)
        self.assertGreater(output.shape[1], 0)


class TestCPCVectorizer(unittest.TestCase):

    def test__can_encode_one(self):
        cpc = "H04W52/02"
        output = CPCVectorizer().embed(cpc)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 1)
        self.assertGreater(output.shape[0], 0)

    def test__can_encode_multiple(self):
        cpcs = ["H04W52/02", "H04W72/00"]
        output = CPCVectorizer().encode_many(cpcs)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output.shape), 2)
        self.assertGreater(output.shape[1], 0)

if __name__ == "__main__":
    unittest.main()
