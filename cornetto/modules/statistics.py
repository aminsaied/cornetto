#!/usr/bin/env python3
"""Conducts statistical analysis of our text data.

Various concepts from information theory are implemented:
    - entropy
    - mutual information
    - information gained
"""
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from containers import Vocab
from text_processor import WordSelector

class IG(object):
    """Information gained."""
    def __init__(self, vocab, docs, labels):
        self.label_dict = self._get_label_dict(labels)
        docs_as_words = IG._docs_to_words(vocab, docs)
        self.scores = self._scores_dict(vocab, docs_as_words, labels)

    @classmethod
    def from_dataframe(cls, vocab, df):
        cols = df.columns.values
        NUMBER_OF_COLS = 2
        assert cols.size == NUMBER_OF_COLS
        aslist = lambda col_name: df[col_name].tolist()
        return cls(vocab, aslist(cols[0]), aslist(cols[1]))

    @staticmethod
    def _docs_to_words(vocab, docs):
        word_selector = WordSelector(vocab)
        select_words = lambda s: word_selector.select_words(s)
        docs_as_words = list(map(select_words, docs))
        return docs_as_words

    def _get_label_dict(self, labels):
        label_list = list(set(labels))
        indices = range(len(label_list))
        label_dict = dict(zip(label_list, indices))
        return label_dict

    def _scores_dict(self, vocab, docs_as_words, labels):
        vocab_scores = self._compute_scores(vocab, docs_as_words, labels)
        return dict(zip(vocab.sorted_keys, vocab_scores))

    def _compute_scores(self, vocab, docs_as_words, labels):
        # array P(label)
        P_label = self._compute_P_label(labels)

        # P(word) and P(word|label)
        P_w, P_w_label = self._compute_P_w_and_P_w_label(vocab, docs_as_words, labels)

        # P(label|word)
        P_label_w = self._compute_P_label_w(P_label, P_w, P_w_label)

        scores = IG.entropy(P_label) - IG.entropy(P_label_w, axis=1)
        return scores

    def _compute_P_label(self, labels):
        N = len(self.label_dict)
        count = np.zeros(N)
        for label in labels:
            label_index = self.label_dict[label]
            count[label_index] += 1
        SMOOTHING = 1.0
        P_label = (count+SMOOTHING)/(len(labels)+N)
        return P_label

    def _compute_P_w_and_P_w_label(self, vocab, docs_as_words, labels):
        count_matrix = np.zeros( (len(vocab),len(self.label_dict)) )
        total_words = 0
        for i, doc in enumerate(docs_as_words):
            label_index = self.label_dict[labels[i]]
            for word in doc:
                word_index = vocab[word].id
                count_matrix[word_index][label_index] += 1
                total_words += 1

        # this works
        SMOOTHING = 1.0
        P_w = np.sum(count_matrix+SMOOTHING,axis=1)/(total_words+count_matrix.size)

        # this works
        col_sums = np.sum(count_matrix,axis=0)
        P_w_label = (count_matrix+SMOOTHING)/(col_sums[np.newaxis,:]+len(vocab))

        return P_w, P_w_label

    def _compute_P_label_w(self, P_label, P_w, P_w_label):
        # P(label|word)
        # for some reason this gives me worng answer: the resulting matrix
        # has rows not probabilities...
        X = np.multiply(P_w_label,P_label[np.newaxis,:])
        P_label_w = np.divide(X,P_w[:,np.newaxis])
        return P_label_w

    @staticmethod
    def entropy(P,axis=0):
        """
        Given a probability distribution (array P) of
        a random variable X, compute the entropy of X.
        """
        print(np.sum(P,axis=axis))
        return -np.sum( np.multiply(P,np.log2(P)), axis=axis )
