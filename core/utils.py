"""
Utility methods
"""

import re
import numpy as np

def get_sentences(text):
    """Split a given (English) text (possibly multiline) into sentences.
    
    Args:
        text (str): Text to be split into sentences.
    
    Returns:
        list: Sentences.
    """
    sentences = []
    paragraphs = get_paragraphs(text)
    ends = r"\b(etc|viz|fig|FIG|Fig|e\.g|i\.e|Nos|Vol|Jan|Feb|Mar|Apr|\
    Jun|Jul|Aug|Sep|Oct|Nov|Dec|Ser|Pat|no|No|Mr|pg|Pg|figs|FIGS|Figs)$"
    for paragraph in paragraphs:
        chunks = re.split(r"\.\s+", paragraph)
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if re.search(ends, chunk) and i < len(chunks)-1:
                chunks[i] = chunk + '. ' + chunks[i+1]
                chunks.pop(i+1)
            elif i < len(chunks)-1:
                chunks[i] = chunks[i] + '.'
            i += 1
        for sentence in chunks:
            sentences.append(sentence)
    return sentences


def get_paragraphs(text):
    r"""Split a text into paragraphs. Assumes paragraphs are separated
    by new line characters (\n).
    
    Args:
        text (str): Text to be split into paragraphs.
    
    Returns:
        list: Paragraphs.
    """
    return [s.strip() for s in re.split("\n+", text) if s.strip()]

def is_cpc_code(item):
    """Check if an item is a Cooperative Patent Classification code.
    Should also work for IPC codes because they have same format.

    Examples:
    H04W52/00 => True
    H04W => False
    H04W005202 => False

    Args:
        item (str): String to be checked.

    Returns:
        bool: True if input string is a CPC code, False otherwise.
    """
    if not isinstance(item, str):
        return False
    pattern = r"^[ABCDEFGHY]\d\d[A-Z]\d+\/\d+$"
    return bool(re.fullmatch(pattern, item))

def normalize_rows(M):
    return normalize_along_axis(M, 1)

def normalize_cols(M):
    return normalize_along_axis(M, 0)

def normalize_along_axis(M, axis):
    epsilon = np.finfo(float).eps
    norms = np.sqrt((M * M).sum(axis=axis, keepdims=True))
    norms += epsilon  # to avoid division by zero
    return M / norms

class Singleton(type):
    """
    This is Singleton metaclass for making Singleton Classes
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
