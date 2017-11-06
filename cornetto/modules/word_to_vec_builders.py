import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import word2vec

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import data
from text_processor import WordSelector

class WordToVecModel(object):
    """
    Handles the word2vec dictionary.
    """
    
    PATH = os.path.dirname(data.__file__)+'/models/w2v/'
    
    def __init__(self, w2v_dict=dict(), dim=100):
        self._w2v_dict = w2v_dict
        self.dim = dim

    def __getitem__(self, word):
        return self._w2v_dict[word]

    def __len__(self):
        return len(self._w2v_dict)

    def __contains__(self, word):
        return (word in self._w2v_dict)

    def save(self, filename):
        EXT = ".pkl"
        data = (self._w2v_dict, self.dim)
        pd.to_pickle(data, self.__class__.PATH+filename+EXT)

    @classmethod
    def load(cls, filename):
        EXT = ".pkl"
        data = pd.read_pickle(cls.PATH+filename+EXT)
        return cls(*data)

class GensimWordToVec(WordToVecModel):
    """Simple wrapper for the Gensim WordToVec model."""
        
    def fit(self, sentences, vocab, window=20):
        # TODO: don't we assume sentences are already built out of words from vocab?
        # so maybe we can omit vocab as input
        
        # works for either pd.Series or just a list
        sentences = list(sentences)
        model = word2vec.Word2Vec(sentences, size=self.dim, window=window)
        words = [word for word in vocab if word in model]
        vectors = [model[word] for word in words]
        self._w2v_dict = dict(zip(words, vectors))
        
class SVDWordToVec(WordToVecModel):
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
