import numpy as np
import pandas as pd
from gensim.models import word2vec

from text_processor import WordSelector
from data_handlers import WordToVecModel


class WordToVecDict(object):
    """Parent class constructing word2vec models."""
    def __init__(self):
        pass

    def build(self):
        """Retuns a WordToVecModel."""
        words = self.vocab.sorted_keys

        if self.model:
            vectors = [self._convert(word) for word in words if
                      (word in self.vocab and word in self.model) ]
        else:
            vectors = [self._convert(word) for word in words if
                       (word in self.vocab) ]

        d = dict(zip(words, vectors))
        return WordToVecModel(d)

    def _convert(self, word):
        pass

class GensimWordToVecBuilder(WordToVecDict):
    """Simple wrapper for the Gensim WordToVec model."""
    def __init__(self, series, vocab, size=100, window=20):
        """
        -- series: pandas Series, containing lists of words
        """
        sentences = series.tolist()
        self.model = word2vec.Word2Vec(sentences)

    def _convert(self, word):
        return self.model[word]

class NaiveWordToVec(WordToVecDict):
    def __init__(self, vocab):
        self.vocab = vocab

    def _convert(self, text):
        """
        Accepts a text, returns the corresponding one-hot vector.
        """
        word_selector = WordSelector(self.vocab)
        words = word_selector.select_words(text)

        vec = np.zeros((len(self.vocab),1))
        for word in words:
            vec[self.vocab[word].id] += 1.
        return vec

class SVDWordToVec(WordToVecDict):
    """
    Uses SVD to produce word embeddings.
    """

    def __init__(self, dim, vocab):
        """
        -- dim: int, the desired dimension of word embeddings
        -- vocab, Vocab
        """
        self.dim = dim
        self.vocab = vocab
        self.index = lambda word: self.vocab[word].id

    def from_series(self, texts):
        """
        Given a list of abstracts, computes the SVD of the
        matrix of word embeddings.
        -- texts: list, of abstracts (as strings)
        """
        A = np.zeros((len(self.vocab), len(texts)))
        word_selector = WordSelector(vocab)
        for text_index, text in enumerate(texts):
            words = word_selector.select_words(text)
            for word in words:
                A[self.index(word)][text_index] += 1
        U, __, __ = np.linalg.svd(A)
        self.word_vectors = U[:, :self.dim]

    def _convert(self, word):
        index = self.index(word)
        return self.word_vectors[index]
