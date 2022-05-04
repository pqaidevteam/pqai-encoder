import unittest
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

env_file = str((Path(__file__).parent.parent / '.env').resolve())
load_dotenv(env_file)

PROTOCOL = 'http'
HOST = 'localhost'
PORT = os.environ['PORT']
API_ENDPOINT = "{}://{}:{}".format(PROTOCOL, HOST, PORT)

class TestAPI(unittest.TestCase):

	def setUp(self):
		pass

	def test_encode_route(self):
		url = "{}/encode?text=sometext".format(API_ENDPOINT)
		response = requests.get(url)
		self.assertEqual(200, response.status_code)


if __name__ == '__main__':
	unittest.main()