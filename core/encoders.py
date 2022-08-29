"""
This module defines encoders, which transform data into representations
suitable as inputs for machine learning pipelines.
"""

import re
from pathlib import Path
import numpy as np
from core.utils import get_sentences
from core.representations import BagOfEntities

BASE_DIR = str(Path(__file__).parent.parent.resolve())
ASSETS_DIR = f"{BASE_DIR}/assets"


class Encoder:

    """An abstract encoder"""

    def encoder_fn(self, data):
        """Encoder function. This is to be defined in concrete implementations
            of an encoder, but should not be directly called while encoding.
            Instead, the `encode` function should be called.
        """
        raise NotImplementedError

    def is_valid_input(self, data):
        """Validate that the given `data` is a compatible input for this
            encoder. This function should be defined in a concrete
            implementation of an encoder, but need not be called explicitly
            while encoding. That's because when called `encode`, this check
            is made automatically.
        """
        raise NotImplementedError

    def encode(self, item):
        """Return a representation of the given item. This function is to be
            used for actual encoding operation, for it includes a validation
            check on the input.
        """
        if not self.is_valid_input(item):
            raise Exception(f"{self.__class__} cannot encode {type(item)}")
        return self.encoder_fn(item)

    def encode_many(self, items):
        """Return representations of the given array of items. This method can
            be overridden whereever there is a way to speed up the encoding
            beyond mere list comprehension, e.g., using some form of parallel or
            batch processing technique.

            The returned representations should have some information (explicit
            or implicit) that maps input to outputs. For example, returning the
            representations in the same order as the inputs, or returning a
            dictionary with input:representation as key:value pairs.
        """
        return [self.encode(item) for item in items]


class TextEncoder(Encoder):

    """Abstract class for encoders that create vector representations"""

    def is_valid_input(self, item):
        """Validate that the item is encodable"""
        return isinstance(item, str)

    def encode_many(self, items):
        """Encode multiple items in one go"""
        return np.array([self.encode(item) for item in items])

    def encoder_fn(self, data):
        """Encoder function"""
        raise NotImplementedError


class BagOfEntitiesEncoder(TextEncoder):

    """Converts a piece of text into a set (bag) of entities"""

    def __init__(self, vocab: list):
        """Initialize"""
        super().__init__()
        self._vocab = vocab
        self._lut = set(vocab)  # look up table
        self._cased = False  # case sensitive when True
        self._sep = " "  # separator
        self._maxlen = 3  # no. of words in longest entity
        self._no_overlap = True

    def encoder_fn(self, text: str):
        """Encode given `text` as a bag of entities"""
        entities = []
        for sent in get_sentences(text):
            entities += self._get_entities_from_sentence(sent)
        if self._no_overlap:
            entities = BagOfEntities(entities).non_overlapping()
        return entities

    def _get_entities_from_sentence(self, sentence: str):
        """Extract entities from a given sentence"""
        candidates = self._get_candidate_entities(sentence)
        return [c for c in candidates if c in self._lut]

    def _get_candidate_entities(self, sent):
        """Extract potential entity candidates (some of the candidates may not
            make sense but they will be filtered out later)
        """
        candidates = set()
        tokens = self._tokenize(sent)
        for n in range(1, self._maxlen + 1):
            for n_gram in self._get_n_grams(n, tokens):
                candidates.add(n_gram)
        return candidates

    def _get_n_grams(self, n: int, tokens: list):
        """Return all possible 1, 2, ..., n-grams created from given `tokens`"""
        if len(tokens) < n:
            return []
        sep = self._sep
        n_grams = [sep.join(tokens[i : i + n]) for i in range(len(tokens))]
        return n_grams

    def _tokenize(self, text: str):
        """Split `text` into words"""
        text = text if self._cased else text.lower()
        pattern = r"([\w\-]+|\W+)"
        matches = re.findall(pattern, text)
        tokens = [m for m in matches if m.strip()]
        return tokens

    @classmethod
    def from_vocab_file(cls, vocab_file: str, blklst_file: str = None):
        """Instantiate from a text file containing entities (one per line)"""
        blacklist = set()
        if blklst_file:
            blacklist = set(cls._read_vocab(blklst_file))
        vocab = cls._read_vocab(vocab_file)
        vocab = [e for e in vocab if e not in blacklist]
        return BagOfEntitiesEncoder(vocab)

    @staticmethod
    def _read_vocab(file: str):
        """"Read entities from a text file (one entity per line)"""
        with open(file, "r") as f:
            vocab = f.read().strip().splitlines()
        return vocab
