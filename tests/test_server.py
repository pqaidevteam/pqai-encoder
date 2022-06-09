import unittest
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

env_file = str((Path(__file__).parent.parent / ".env").resolve())
load_dotenv(env_file)

PROTOCOL = "http"
HOST = "localhost"
PORT = os.environ["PORT"]
API_ENDPOINT = "{}://{}:{}".format(PROTOCOL, HOST, PORT)


class TestAPI(unittest.TestCase):
    
    def setUp(self):
        pass

    def test__can_encode_text(self):
        text = "Some random text"
        url = "{}/encode".format(API_ENDPOINT)
        payload = {"data": text, "encoder": 'sbert'}
        response = requests.post(url, json=payload)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertIsInstance(data["vector"], list)

    def test__throws_error_when_no_encoder_specified(self):
        pass

if __name__ == "__main__":
    unittest.main()
