import re, json
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

from core.utils import Singleton
from core.encoders import Encoder

BASE_DIR = str(Path(__file__).parent.parent.resolve())
models_dir = "{}/assets".format(BASE_DIR)


class Vectorizer(Encoder):
    def __init__(self):
        super().__init__()
        self._name = "Vectorizer"
        self._encoder_fn = self.embed

    def _input_validation_fn(self, item):
        return isinstance(item, str)

    def embed(self, item):
        pass

    def encode_many(self, items):
        return np.array([self.embed(item) for item in items])


class SentBERTVectorizer(Vectorizer, metaclass=Singleton):

    sentbert_model_path = models_dir.rstrip("/") + "/vectorizer_distilbert_poc/"

    def __init__(self):
        super().__init__()
        self._name = "SentBERTVectorizer"
        self._model = None  # Lazy loads

    def load(self):
        self._model = SentenceTransformer(self.sentbert_model_path)

    def embed(self, text):
        self._load_if_needed()
        vec = self._model.encode([text])[0]
        return vec

    def encode_many(self, texts):
        self._load_if_needed()
        vecs = np.array(self._model.encode(texts))
        return vecs

    def _load_if_needed(self):
        if self._model is None:
            self.load()


class CPCVectorizer(Vectorizer, metaclass=Singleton):

    cpc_list_file = models_dir.rstrip("/") + "/cpc_vectors_256d.items.json"
    cpc_vecs_file = models_dir.rstrip("/") + "/cpc_vectors_256d.npy"

    def __init__(self):
        super().__init__()
        self._name = "CPCVectorizer"
        with open(self.cpc_list_file) as file:
            self.vocab = json.load(file)
        self.lut = {cpc: i for (i, cpc) in enumerate(self.vocab)}
        self.vecs = np.load(self.cpc_vecs_file)
        self.dims = self.vecs.shape[1]
        self.gray = 0.00001 * np.ones(self.dims)

    def __getitem__(self, cpc_code):
        if cpc_code not in self.lut:
            return np.zeros(self.dims)
        i = self.lut[cpc_code]
        return self.vecs[i]

    def embed(self, cpcs):
        if not [cpc for cpc in cpcs if cpc in self.lut]:
            return self.gray
        cpc_vecs = [self[cpc] for cpc in cpcs if cpc in self.lut]
        avg_cpc_vec = np.average(np.array(cpc_vecs), axis=0)
        return avg_cpc_vec


class SIFTextVectorizer(Vectorizer, metaclass=Singleton):

    word_vecs_file = models_dir.rstrip("/") + "/glove-We.npy"
    word_list_file = models_dir.rstrip("/") + "/glove-vocab.json"
    word_freq_file = models_dir.rstrip("/") + "/dfs.json"

    def __init__(self):
        super().__init__()
        self._name = "SIFTextVectorizer"

        self.alpha = 0.015
        self.vocab = self._read_json(self.word_list_file)
        self.dfs = self._read_json(self.word_freq_file)
        self.vecs = np.load(self.word_vecs_file)
        self.lut = self._lookup_table()
        self.sifs = [self._sif(w) for w in self.vocab]
        self.dims = self.vecs.shape[1]
        self.gray = 0.00001 * np.ones(self.dims)

    def _read_json(self, filepath):
        with open(filepath) as file:
            return json.load(file)

    def _lookup_table(self):
        return {cpc: i for (i, cpc) in enumerate(self.vocab)}

    def _sif(self, word):
        if not word in self.dfs:
            return 1.0
        df = self.dfs[word]
        df_max = self.dfs["the"] + 1
        proba = df / df_max
        return self.alpha / (self.alpha + proba)

    def __getitem__(self, word):
        if word not in self.lut:
            return np.zeros(self.dims)
        i = self.lut[word]
        return self.vecs[i]

    def tokenize(self, text):
        words = re.findall(r"\w+", text.lower())
        return words if words is not None else []

    def embed(self, text, unique=True, remove_pc=False, average=False):
        words = self.tokenize(text)
        if not words:
            return self.gray
        if unique:
            words = list(set(words))
        idxs = [self.lut[w] for w in words if w in self.lut]
        if not idxs:
            return self.gray
        if not average:
            matrix = np.array([self.vecs[i] * self.sifs[i] for i in idxs])
        else:
            matrix = np.array([self.vecs[i] for i in idxs])
        if remove_pc:
            matrix = self.remove_first_pc(matrix)
        vec = np.average(matrix, axis=0)
        return vec

    def remove_first_pc(self, X):
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(X)
        pc = svd.components_
        X = X - X.dot(pc.transpose()) * pc
        return X


class EmbeddingMatrix:

    """An EmbeddingMatrix is a dictionary-like structure, where keys are item
        identifiers (labels) and values are their vectors (embeddings).
    """

    def __init__(self, labels, vectors):
        self._labels = labels
        self._lut = {w: i for i, w in enumerate(self._labels)}
        self._vectors = vectors
        self._unit_vectors = self._create_unit_vectors()

    @property
    def dims(self):
        vec = self._vectors[0]
        return len(vec)

    def __getitem__(self, item):
        idx = self._lut[item]
        return self._vectors[idx]

    def __contains__(self, item):
        return item in self._lut

    def similar_to_item(self, item, n=10, dist="cosine"):
        idx = self._lut[item]
        vector = self._unit_vectors[idx]
        return self.similar_to_vector(vector, n)

    def similar_to_vector(self, vector, n=10, dist="cosine"):
        if dist == "cosine":
            dists = self._cosine_dists(vector, self._unit_vectors)
        elif dist == "euclidean":
            dists = self._euclid_dists(vector, self._vectors)
        elif dist == "dot":
            dists = self._dot_prods(vector, self._vectors)
        idxs = np.argsort(dists)[:n]
        return [self._labels[i] for i in idxs]

    def _euclid_dists(self, a, b):
        d = a - b
        return np.sum(d * d, axis=1)

    def _cosine_dists(self, a, b):
        return 1 - np.dot(a, b.T)

    def _dot_prods(self, a, b):
        return -np.dot(a, b.T)

    def _create_unit_vectors(self):
        return normalize_rows(self._vectors)

    @classmethod
    def from_txt_npy(cls, txt_filepath, npy_filepath):
        """Create an `EmbeddingMatrix` from an items file containing the
        a list of item descriptions (one per line) and a numpy file with
        the vectors that have one-to-one correspondance with the items.

        Args:
            txt_filepath (str): Path to items file
            npy_filepath (str): Path to numpy (vectors) file

        Returns:
            EmbeddingMatrix: Resulting embedding matrix object
        """
        with open(txt_filepath) as file:
            items = [l.strip() for l in file if l.strip()]
        vectors = np.load(npy_filepath)
        return EmbeddingMatrix(items, vectors)

    @classmethod
    def from_tsv(cls, filepath):
        """Create an `EmbeddingMatrix` from a tsv file where the first
                column contains the item descriptions and subsequent columns
                contain the vector components. All columns should be separated
        by single tabs.

                Args:
                    filepath (str): Path to tsv (tab separated values) file

                Returns:
                    EmbeddingMatrix: Resulting embedding matrix object
        """
        pairs = cls._parse_tsv_file(filepath)
        items = [word for word, _ in pairs]
        vectors = np.array([vector for _, vector in pairs])
        return EmbeddingMatrix(items, vectors)

    @classmethod
    def _parse_tsv_file(cls, filepath):
        with open(filepath) as file:
            lines = (l for l in file if l.strip())
            pairs = [cls._parse_tsv_line(l) for l in lines]
        return pairs

    @classmethod
    def _parse_tsv_line(cls, line):
        [word, *vector] = line.strip().split("\t")
        vector = [float(val) for val in vector]
        return word, vector


class BagOfVectorsEncoder(Encoder):
    """
    This class is a bag of words encoder class extending Encoder class.


    """

    def __init__(self, emb_matrix):
        super().__init__()
        self._emb_matrix = emb_matrix
        self.encoder_fn = self._vectorize_items
        self.is_valid_input = lambda x: isinstance(x, list)

    def _vectorize_items(self, bag_of_items):
        items = [item for item in bag_of_items if item in self._emb_matrix]
        vectors = [self._emb_matrix[item] for item in items]
        vectors_as_tuples = [tuple(vec) for vec in vectors]
        return set(vectors_as_tuples)

    @classmethod
    def from_txt_npy(cls, txtfile, npyfile):
        emb_matrix = EmbeddingMatrix.from_txt_npy(txtfile, npyfile)
        return BagOfVectorsEncoder(emb_matrix)