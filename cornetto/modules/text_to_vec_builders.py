import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from text_processor import WordSelector

class TextToVecModel(object):
    """Parent class with convert method."""
    def __init__(self):
        pass

    def convert(self, text):
        pass

class T2VFromW2V(TextToVecModel):
    """
    Given a word2vec model and a number of words, converts sentences to
    vectors by stacking the word2vec representations.
    """
    def __init__(self, w2v_model, N_words=25):
        """
        -- w2v_model: dictionary, keys=words values=word2vec representations
        -- N_words: int (default=25), number of words selected
        """
        self._N_words = N_words
        self._w2v_model = w2v_model
        self.dim = w2v_model.dim * N_words

    def convert(self, text):
        """
        Stacks word2vec representations into one long vector.
        -- text: string
        """
        word_selector = WordSelector(self._w2v_model)
        words = word_selector.select_N_words(text, N_words = self._N_words)
        vecs = [self._w2v_model[word] for word in words]
        return self._assemble_vecs(vecs)

    def _assemble_vecs(self, vecs):
        if len(vecs) == self._N_words:
            return np.array(vecs).flatten()
        else:
            return np.zeros(self.dim)

    def save(self, filename):
        EXTENSION = '.pkl'
        data = [self._w2v_model,self._N_words]
        pd.to_pickle(data, filename+EXTENSION)

    @classmethod
    def load(cls, filename):
        EXTENSION = '.pkl'
        try:
            data = pd.read_pickle(filename+EXTENSION)
        except FileNotFoundError:
            print("No file by that name. Remember, we add the .pkl extension for you!")
        return cls(*data)


class T2VFromVocab(TextToVecModel):
    """
    Given a vocab object, converts sentences into their one-hot representations.
    """
    def __init__(self, vocab):
        """
        Accepts a pre-trained Vocab (or TFIDF) object.
        -- vocab: Vocab
        """
        self.vocab = vocab

    def convert(self, text):
        """
        Accepts a text, returns the corresponding one-hot vector.
        -- text: string
        """
        word_selector = WordSelector(self.vocab)
        words = word_selector.select_words(text)

        vec_normal = self._convert_sentence(words)
        return vec_normal

    def _convert_sentence(self, sentence):
        """
        Given a sentence (=list of words), return the corresponding one-hot vector.
        -- sentence: list, of words
        """
        return self.vocab.one_hot(sentence)
