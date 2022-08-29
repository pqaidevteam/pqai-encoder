[![Python](https://img.shields.io/badge/python-v3.10-blue)](https://www.python.org/)
[![Linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Docker build: automated](https://img.shields.io/badge/docker%20build-automated-066da5)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

_Note: This repository is under activity development and not ready for production yet._

# PQAI Encoder (WIP)

PQAI service for transforming data into representations suitable as inputs for machine-learning pipelines. For example, transforming a piece of text into an embedding in a high-dimensional vector space.

The following representations are supported at the moment:

1. Text -> Dense embedding
1. Text -> Sequence of word vectors
1. Text -> Bag of entities

## Routes

| Method | Route     | Description                            |
| ------ | --------- | -------------------------------------- |
| `POST` | `/encode` | Returns a representation of given data |


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