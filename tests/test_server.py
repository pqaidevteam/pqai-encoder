import unittest
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi.testclient import TestClient

BASE_PATH = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_PATH / ".env"

sys.path.append(BASE_PATH.as_posix())
load_dotenv(ENV_PATH.as_posix())

from main import app


class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
        self.client.testing = True

    def test__can_vectorize_text_single(self):
        for encoder in ["sbert", "sif"]:
            text = "Some random text"
            payload = {"data": text, "encoder": encoder}
            response = self.client.post("/encode", json=payload)
            self.assertEqual(200, response.status_code)
            data = response.json()
            self.assertEqual(data["original"], text)
            self.assertIsInstance(data["encoded"], list)
            self.assertTrue(all(isinstance(x, float) for x in data["encoded"]))

    def test__can_vectorize_text_multiple(self):
        for encoder in ["sbert", "sif"]:
            texts = ["Some random text", "Another random piece of text"]
            payload = {"data": texts, "encoder": encoder}
            response = self.client.post("/encode", json=payload)
            self.assertEqual(200, response.status_code)
            data = response.json()
            self.assertEqual(data["original"], texts)
            self.assertIsInstance(data["encoded"], list)
            self.assertTrue(all(isinstance(x, list) for x in data["encoded"]))

    def test__can_create_bag_of_entities(self):
        text = "The present invention describes a computer for playing games."
        payload = {"data": text, "encoder": "boe"}
        response = self.client.post("/encode", json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(data["original"], text)
        self.assertIsInstance(data["encoded"], list)
        self.assertTrue(all(isinstance(x, str) for x in data["encoded"]))

    def test__can_return_embedding_for_one_entity(self):
        entity = "computer"
        payload = {"data": entity, "encoder": "emb"}
        response = self.client.post("/encode", json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(data["original"], entity)
        self.assertIsInstance(data["encoded"], list)
        self.assertTrue(all(isinstance(x, float) for x in data["encoded"]))

    def test__can_return_embedding_for_multiple_entities(self):
        entities = ["computer", "games"]
        payload = {"data": entities, "encoder": "emb"}
        response = self.client.post("/encode", json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(data["original"], entities)
        self.assertIsInstance(data["encoded"], list)
        self.assertTrue(all(isinstance(x, list) for x in data["encoded"]))

    def test__throws_error_when_no_encoder_specified(self):
        text = "Some random text"
        payload = {"data": text}  # no encoder parameter!
        response = self.client.post("/encode", json=payload)
        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
