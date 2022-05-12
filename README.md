# PQAI Encoder (WIP)

PQAI service for creating representations of items. Items can be things like text or patent classification codes (e.g. CPC). Representations can be high-dimensional embeddings, bag of words, etc.

## License

The project is open-source under the MIT license.

## Contribute

We welcome contributions.

To make a contribution, please follow these steps:

1. Fork this repository.
2. Create a new branch with a descriptive name
3. Make copy of env file as .env and docker-compose.dev.yml as docker-compose.yml
4. Download and extract the file from `https://s3.amazonaws.com/pqai.s3/public/pqai-assets-latest.zip` to assets/
4. Bring encoder to life `docker-compose up`
5. Make the changes you want and add new tests, if needed
6. Make sure all tests are passing `docker exec -i dev_pqai_encoder_api python -m unittest discover ./tests/`
7. Commit your changes
8. Submit a pull request

## Support

Please create an issue if you need help.