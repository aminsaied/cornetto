#!/usr/bin/env python3
"""Implements the naive Bayes algorithm to classify text.

This model serves as a baseline against which to compare more
sophisticated deep-learning models.

Can be run as a script that will build and train this model.
"""
import numpy as np
import pandas as pd
from data_handlers import UnigramTrainingData

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from containers import Vocab, MSC
from prepare_data import MSCCleaner

EXT = '.pkl'
VOCAB_POSTFIX = '_vocab'
MSC_POSTFIX = '_msc'
DELIMITER = "|"

class Unigram(object):
    """
    This class implements the "Unigram model of language". Namely, it assumes
    that text is generated by doing independent sampling from a distribution
    on the set of all words. Given the training data as pairs (text, class),
    the method 'fit' approximates the probability distributions for each of
    the classes. The 'predict' method uses Naive Bayes algorithm to compute
    the probability of a given text belonging to each of the possible classes
    (labels).
    """
    def __init__(self, vocab, msc_bank, topics=None):
        """
        Input:
          -- vocab : object of Vocab, 'bag of words'
          -- msc_bank : object of MSC, contains the classes
          -- topics : 2D array, len(msc_bank)-by-len(vocab),
          each row = probability distribution
        """
        ### TODO: replace msc_bank with any set of labels.
        self.vocab = vocab
        self.msc_bank = msc_bank
        self.topics = topics

    def save(self, filename):
        """
        Input:
          -- filename: str, *no extension*,
          i.e. 'my_file' isnstead of 'my_file.txt'
        """
        self.vocab.save(filename+VOCAB_POSTFIX)
        self.msc_bank.save(filename+MSC_POSTFIX)
        np.savetxt(filename+EXT, self.topics, delimiter=DELIMITER)

    @classmethod
    def load(cls, filename):
        vocab = Vocab.load(filename+VOCAB_POSTFIX)
        msc_bank = MSC.load(filename+MSC_POSTFIX)
        topics = np.loadtxt(filename+EXT, delimiter=DELIMITER)
        return cls(vocab, msc_bank, topics)

    def train(self, data):
        '''
        -- data: UnigramTrainingData object
        '''
        self.topics = self._build_topics_matrix(data.training)
        print("Training complete.")

    def _build_topics_matrix(self, training_data):
        abstracts, codes = training_data.X.tolist(), training_data.Y.tolist()
        M = np.zeros((len(self.msc_bank), len(self.vocab)))
        for abstract, list_of_codes in zip(abstracts, codes):
            for code in list_of_codes:
                try:
                    msc_ind = self.msc_bank[code].id
                    for word in abstract:
                        vocab_ind = self.vocab[word].id
                        M[msc_ind][vocab_ind] += 1
                except (KeyError, TypeError, IndexError):
                    pass

        M += np.ones(M.shape)

        M_counts = M.sum(axis=1)
        return M/M_counts[:,np.newaxis]

    def predict(self, sentence):
        """
        Given a sentence, returns a vector of probabilities it belongs to
        each of the classes.
        Input:
          -- sentence: List[str], list of words
        Output:
          -- np.array, probability vector, i-th entry = probability the
          sentence belongs to the i-th class.
        """
        as_vec = self.vocab.one_hot(sentence)
        model_output = np.matmul(self.topics, X)
        model_output_as_prob = model_output/np.sum(model_output)
        return model_output_as_prob


if __name__ == "__main__":

    DEPTH = 2
    FILENAME = '-unigram_model_DEPTH' + str(DEPTH)
    TRAINING_DATA_DIR = '-unigram_training_data_DEPTH' + str(DEPTH)

    arxiv = pd.read_pickle('-arxiv_processed.pkl')
    vocab = Vocab.load('-vocab')
    msc_bank = MSC.load(DEPTH)

    X = arxiv['Abstract']
    Y = MSCCleaner._specify(arxiv['MSCs'], depth=DEPTH)

    unigram_data = UnigramTrainingData.build(X, Y, vocab, msc_bank)

    # NB. would like to save this training data, but there is some bug.
    # unigram_data.save(TRAINING_DATA_DIR)
    # print ("Unigram training data saved: " + TRAINING_DATA_DIR + EXT)

    print ("Training model...")
    model = Unigram(vocab, msc_bank)
    model.train(unigram_data)

    print ("Saving model...")
    model.save(filename=FILENAME)
    print ("Model saved: " + FILENAME + EXT)
