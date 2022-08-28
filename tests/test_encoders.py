"""Tests for the code in `core/encoders.py`

Attributes:
    BASE_DIR (str): Absolute path to application's base directory
    TEST_DIR (str): Absolute path to the test directory
    ENV_PATH (str): Absolute apth to .env file
"""
import unittest
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
ENV_PATH = "{}/.env".format(BASE_DIR)

load_dotenv(ENV_PATH)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append(BASE_DIR)
from core.encoders import Encoder, BagOfEntitiesEncoder


class Text2CharTestEncoder(Encoder):

    """A dummy implementation of `Encoder` for testing
    """

    def encoder_fn(self, string: str):
        """Concrete implementation of the encoding function"""
        return list(string)

    def is_valid_input(self, string: str):
        """Check if the given input is encodable"""
        return isinstance(string, str)


class TestEncoderClass(unittest.TestCase):

    """Test the inherent functionality of abstract `Encoder`

    Attributes:
        encoder (`Text2CharTestEncoder`): Dummy encoder
    """

    def setUp(self):
        """Instantiate an concrete `Encoder` for testing"""
        self.encoder = Text2CharTestEncoder()

    def test__can_create_working_concrete_encoder_from_abstract(self):
        """Should return valid encoded data"""
        data = "hello"
        expected = ["h", "e", "l", "l", "o"]
        actual = self.encoder.encode(data)
        self.assertEqual(expected, actual)

    def test__can_encode_multiple_items(self):
        """Should return valid encoded data for each input item
        """
        data = ["hi", "hello"]
        expected = [list("hi"), list("hello")]
        actual = self.encoder.encode_many(data)
        self.assertEqual(expected, actual)

    def test__raises_exception_on_invalid_input_data(self):
        """Should raise error if an integer or list is given for encoding
        """
        attempt = lambda: self.encoder.encode(1)
        self.assertRaises(Exception, attempt)
        attempt = lambda: self.encoder.encode(["something"])
        self.assertRaises(Exception, attempt)


class TestBagOfEntitiesEncoder(unittest.TestCase):

    """Test functionality of `BagOfEntitiesEncoder`

    Attributes:
        encoder (`BagOfEntitiesEncoder`): An instance of encoder to be tested
    """

    def setUp(self):
        """Instantiate an encoder from test vocab file
        """
        file = f"{TEST_DIR}/test_entities.vocab"
        self.encoder = BagOfEntitiesEncoder.from_vocab_file(file)

    def test__can_encode_a_sentence(self):
        """Should be able to extract entities from a given sentence
        """
        sent = "base station and mobile station"
        expected = set(["base station", "mobile station"])
        actual = self.encoder.encode(sent)
        self.assertEqual(expected, actual)

    def test__can_encode_multi_sentence_text(self):
        """Should respect sentence boundaries when extracting entities
        """
        sents = "Base station and mobile. Station is there."
        expected = set(["base station"])
        actual = self.encoder.encode(sents)
        self.assertEqual(expected, actual)

    def test__can_encode_a_string_devoid_of_entities(self):
        """Should return an empty set when there are no entities in given text
        """
        trivial_string = "the of and"
        actual = self.encoder.encode(trivial_string)
        self.assertEqual(set(), actual)

    def test__can_encode_a_degenerate_string(self):
        """Should return empty set when empty string is encoded
        """
        null_string = ""
        actual = self.encoder.encode(null_string)
        self.assertEqual(set(), actual)

    def test__can_capture_hyphenated_entities(self):
        """Should respect hyphens, e.g., x-ray
        """
        string = "This is an X-ray machine."
        expected = set(["x-ray"])
        actual = self.encoder.encode(string)
        self.assertEqual(expected, actual)

    def test__error_when_invalid_input(self):
        """Should throw an error when an invalid input is given for encoding
        """
        data = [1, 2, 3]
        attempt = lambda: self.encoder.encode(data)
        self.assertRaises(Exception, attempt)


if __name__ == "__main__":
    unittest.main()
