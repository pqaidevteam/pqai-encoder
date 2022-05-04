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

    def test_encode_route(self):
        text = "Some random text"
        url = "{}/encode?text={}".format(API_ENDPOINT, text)
        response = requests.get(url)
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertIsInstance(data["vector"], list)


if __name__ == "__main__":
    unittest.main()
