import re
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

from core.utils import Singleton, normalize_rows
from core.encoders import Encoder, TextEncoder

BASE_DIR = str(Path(__file__).parent.parent.resolve())
ASSETS_DIR = f"{BASE_DIR}/assets"


class SentBERTVectorizer(TextEncoder, metaclass=Singleton):

    """SBERT Vectorizer"""

    model_path = f"{ASSETS_DIR}/vectorizer_distilbert_poc"

    def __init__(self):
        """Initialize"""
        super().__init__()
        self._model = SentenceTransformer(self.model_path)

    def encoder_fn(self, text: str):
        """Create embedding for given text"""
        return self._model.encode([text])[0]

    def encode_many(self, texts: str):
        """Create embeddings for each of a given array of texts"""
        return self._model.encode(texts)


class CPCVectorizer(TextEncoder, metaclass=Singleton):

    """Returns precomputed vector representations for CPC class codes"""

    labels_file = f"{ASSETS_DIR}/cpc_vectors_256d.items.json"
    vector_file = f"{ASSETS_DIR}/cpc_vectors_256d.npy"

    def __init__(self):
        """Initialize"""
        super().__init__()
        with open(self.labels_file) as f:
            self._vocab = json.load(f)
            self._lut = {cpc: i for (i, cpc) in enumerate(self._vocab)}
        self._vecs = np.load(self.vector_file)
        self._dims = self._vecs.shape[1]
        self._gray = 0.00001 * np.ones(self._dims)

    def encoder_fn(self, cpc: str):
        """Return vector representation of the given cpc code"""
        if cpc not in self._lut:
            return self._gray
        i = self._lut[cpc]
        return self._vecs[i]


class EmbeddingMatrix:

    """An EmbeddingMatrix is a dictionary-like structure, where keys are item
        identifiers (labels) and values are their vectors (embeddings).
    """

    def __init__(self, labels: list, vectors: np.ndarray):
        """Initialize"""
        self._labels = labels
        self._lut = {w: i for i, w in enumerate(labels)}
        self._vectors = vectors
        self._unit_vectors = normalize_rows(self._vectors)
        self._dists = {
            "cosine": self._cosine_dists,
            "euclidean": self._euclid_dists,
            "dot": self._dot_prods
        }

    @property
    def dims(self):
        """Dimensionality of embedding space"""
        vector = self._vectors[0]
        return len(vector)

    def __getitem__(self, item: str):
        """Return embedding for a particular item (label)"""
        idx = self._lut[item]
        return self._vectors[idx]

    def __contains__(self, item):
        """Check whether an embedding with given label is present"""
        return item in self._lut

    def similar_to_item(self, item, n=10, dist="cosine"):
        """Return most similar items to the given item"""
        idx = self._lut[item]
        vector = self._unit_vectors[idx]
        return self.similar_to_vector(vector, n)

    def similar_to_vector(self, vector, n=10, dist="cosine"):
        """Return most similar items to the given vector"""
        dist_fn = self._dists[dist]
        dists = dist_fn(vector, self._vectors)
        idxs = np.argsort(dists)[:n]
        return [self._labels[i] for i in idxs]

    def _euclid_dists(self, a, b):
        """Return euclidean distance between the given vectors
        """
        d = a - b
        return np.sum(d * d, axis=1)

    def _cosine_dists(self, a, b):
        """Return cosine distance between the given vectors
        """
        return 1 - np.dot(a, b.T)

    def _dot_prods(self, a, b):
        """Return dot product between the given vectors
        """
        return -np.dot(a, b.T)

    @classmethod
    def from_txt_npy(cls, txt_file, npy_file):
        """Create an `EmbeddingMatrix` from a labels file (one label per line)
            and a .npy file (a 2d matrix)
        """
        with open(txt_file, "r") as file:
            labels = [l.strip() for l in file if l.strip()]
        vectors = np.load(npy_file)
        return EmbeddingMatrix(labels, vectors)

    @classmethod
    def from_json_npy(cls, json_file, npy_file):
        """Create an `EmbeddingMatrix` from a labels file (one label per line)
            and a .npy file (a 2d matrix)
        """
        with open(json_file, "r") as f:
            labels = json.load(f)
        vectors = np.load(npy_file)
        return EmbeddingMatrix(labels, vectors)

    @classmethod
    def from_tsv(cls, filepath):
        """Create an `EmbeddingMatrix` from a tsv file where the first
            column contains the item descriptions and subsequent columns
            contain the vector components. All columns should be separated
            by single tabs.
        """
        labels = []
        vectors = []
        f = open(filepath, "r")
        for line in f:
            if not line.strip():
                continue
            [label, *vector] = line.strip().split("\t")
            vector = [float(val) for val in vector]
            labels.append(label)
            vectors.append(vector)
        f.close()
        return EmbeddingMatrix(labels, np.array(vectors))


class SIFTextVectorizer(TextEncoder, metaclass=Singleton):

    """Computes embeddings for text using SIF weighted word vectors"""

    labels_file = f"{ASSETS_DIR}/glove-vocab.json"
    vector_file = f"{ASSETS_DIR}/glove-We.npy"
    dfs_file = f"{ASSETS_DIR}/dfs.json"

    def __init__(self):
        """Initialize"""
        super().__init__()
        self.alpha = 0.015
        self.E = EmbeddingMatrix.from_json_npy(self.labels_file, self.vector_file)
        with open(self.dfs_file, "r") as f:
            self.dfs = json.load(f)
            self.sifs = {w: self._sif(w) for w in self.dfs}
        self.gray = 0.00001 * np.ones(self.E.dims)
        self.unique = True
        self.remove_pc = False

    def encoder_fn(self, text: str):
        """Return vector representation of the given text
        """
        words = self._tokenize(text)
        if not words:
            return self.gray
        words = list(set(words)) if self.unique else words
        words = [w for w in words if w in self.E]
        M = np.array([self.E[w]*self.sifs[w] for w in words])
        if self.remove_pc:
            M = self.remove_first_pc(M)
        vec = np.average(M, axis=0)
        return vec

    def _sif(self, word: str):
        """Compute smooth inverse frequence for given word
        """
        if not word in self.dfs:
            return 1.0
        df = self.dfs[word]
        df_max = self.dfs["the"] + 1
        proba = df / df_max
        return self.alpha / (self.alpha + proba)

    def _tokenize(self, text: str):
        """Split text into words
        """
        words = re.findall(r"\w+", text.lower())
        return words if words is not None else []

    def _remove_first_pc(self, X: np.ndarray):
        """Remove first principle component
            Citation: Arora, Sanjeev et al. “A Simple but Tough-to-Beat Baseline
            for Sentence Embeddings.” ICLR (2017).
        """
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(X)
        pc = svd.components_
        X = X - X.dot(pc.transpose()) * pc
        return X


class BagOfVectorsEncoder(Encoder):

    """Bag of vectors encoder"""

    def __init__(self, emb_matrix):
        """Initialize"""
        super().__init__()
        self._emb_matrix = emb_matrix

    def is_valid_input(self, data):
        return isinstance(data, (list, set))

    def encoder_fn(self, bag_of_items):
        items = [item for item in bag_of_items if item in self._emb_matrix]
        vectors = [self._emb_matrix[item] for item in items]
        vectors_as_tuples = [tuple(vec) for vec in vectors]
        return set(vectors_as_tuples)

    @classmethod
    def from_txt_npy(cls, txtfile, npyfile):
        emb_matrix = EmbeddingMatrix.from_txt_npy(txtfile, npyfile)
        return BagOfVectorsEncoder(emb_matrix)
