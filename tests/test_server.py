import unittest
import requests
import os
import socket
from dotenv import load_dotenv
from pathlib import Path

env_file = str((Path(__file__).parent.parent / ".env").resolve())
load_dotenv(env_file)

PROTOCOL = "http"
HOST = "localhost"
PORT = os.environ["PORT"]
API_ENDPOINT = "{}://{}:{}".format(PROTOCOL, HOST, PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_not_running = sock.connect_ex((HOST, int(PORT))) != 0

if server_not_running:
    print("Server is not running. API tests will be skipped.")


@unittest.skipIf(server_not_running, "Works only when true")
class TestAPI(unittest.TestCase):
    def setUp(self):
        self.route = f"{API_ENDPOINT}/encode"

    def test__can_vectorize_text_single(self):
        for encoder in ["sbert", "sif"]:
            text = "Some random text"
            payload = {"data": text, "encoder": encoder}
            response = requests.post(self.route, json=payload)
            self.assertEqual(200, response.status_code)
            data = response.json()
            self.assertEqual(data["original"], text)
            self.assertIsInstance(data["encoded"], list)
            self.assertTrue(all(isinstance(x, float) for x in data["encoded"]))

    def test__can_vectorize_text_multiple(self):
        for encoder in ["sbert", "sif"]:
            texts = ["Some random text", "Another random piece of text"]
            payload = {"data": texts, "encoder": encoder}
            response = requests.post(self.route, json=payload)
            self.assertEqual(200, response.status_code)
            data = response.json()
            self.assertEqual(data["original"], texts)
            self.assertIsInstance(data["encoded"], list)
            self.assertTrue(all(isinstance(x, list) for x in data["encoded"]))

    def test__can_create_bag_of_entities(self):
        text = "The present invention describes a computer for playing games."
        payload = {"data": text, "encoder": "boe"}
        response = requests.post(self.route, json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(data["original"], text)
        self.assertIsInstance(data["encoded"], list)
        self.assertTrue(all(isinstance(x, str) for x in data["encoded"]))

    def test__can_return_embedding_for_one_entity(self):
        entity = "computer"
        payload = {"data": entity, "encoder": "emb"}
        response = requests.post(self.route, json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(data["original"], entity)
        self.assertIsInstance(data["encoded"], list)
        self.assertTrue(all(isinstance(x, float) for x in data["encoded"]))

    def test__can_return_embedding_for_multiple_entities(self):
        entities = ["computer", "games"]
        payload = {"data": entities, "encoder": "emb"}
        response = requests.post(self.route, json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(data["original"], entities)
        self.assertIsInstance(data["encoded"], list)
        self.assertTrue(all(isinstance(x, list) for x in data["encoded"]))

    def test__throws_error_when_no_encoder_specified(self):
        text = "Some random text"
        payload = {"data": text}  # no encoder parameter!
        response = requests.post(self.route, json=payload)
        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
