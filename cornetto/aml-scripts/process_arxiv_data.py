#!/usr/bin/env python3
"""Script processing arxiv data by cleaning text and building a vocabulary.

This script is desined to do two things. First, it processes the arxiv
dataframe collected by 'harvest_arxiv.py'. Namely:

  - removes punctuation and math equations from abstracts
  - keeps only nouns, proper nouns and adjectives
  - learns phrases (2-grams) and joins the words in them by '_'
  - only keeps valid MSC labels, and abstracts with enough words in them

Moreover, while processing the dataframe it learns a vocabulary.
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.container_builders import VocabBuilder as VB
from modules.container_builders import PhraseBuilder as PB
from modules.arxiv_processor import PrepareInput, MSCCleaner
from modules.text_processor import TextCleaner
from modules.datasets import search_files, read_raw_arxiv_data, INPUT_, OUTPUT_

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

ARXIV_DATA_DIR = '../data/arxiv_raw/'
ARXIV_PROC_DIR = '../data/arxiv/'

VOCAB_DIR = '../data/vocab/'
VOCAB_PREFIX = '-vocab_'
PROCESSED_PREFIX = '-processed_'

PHRASE_PERCENT = 0.7
MIN_ABSTR = 10
DEPTH = 5

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Returns series from pandas dataframe."""
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.column]

class AbstractCleaner(BaseEstimator, TransformerMixin):
    """Cleans abstracts of unwanted math expressions."""
    def __init__(self, math_replacement=''):
        self.math_replacement = math_replacement

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        f = lambda abstract: TextCleaner.get_clean_words(abstract)
        return X.apply(f)

class VocabSelector(BaseEstimator, TransformerMixin):
    """Builds vocab from abstracts."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        temp_vocab = VocabSelector._build_vocab(X)
        phrases = VocabSelector._build_phrases(X, temp_vocab)
        phrases_vocab = phrases.to_vocab()

        vocab = temp_vocab + phrases_vocab

        return vocab

    @staticmethod
    def _build_vocab(sentences):
        print("Building the vocabulary.")
        vb = VB()
        vocab = vb.from_sentences(sentences)
        return vocab

    @staticmethod
    def _build_phrases(sentences, vocab):
        print("Building phrases.")
        pb = PB(vocab)
        p = PHRASE_PERCENT
        phrases = pb.from_sentences(sentences, percentage=p)
        return phrases

if __name__ == "__main__":

    filename = search_files(ARXIV_DATA_DIR)
    if not filename:
        sys.exit()

    arxiv = read_raw_arxiv_data(ARXIV_DATA_DIR+filename)

    target_pipeline = Pipeline([
            ('selector'       , DataFrameSelector(OUTPUT_) ),
            ('cleaner'        , MSCCleaner(depth=DEPTH)),
        ])

    msc_series = target_pipeline.fit_transform(arxiv)

    cleaner_pipeline = Pipeline([
            ('selector'  , DataFrameSelector(INPUT_) ),
            ('cleaner'   , AbstractCleaner()),
        ])

    sentences_series = cleaner_pipeline.fit_transform(arxiv)

    vocab_pipeline = Pipeline([
            ('vocab_builder', VocabSelector() ),
    ])

    vocab = vocab_pipeline.fit_transform(sentences_series)
    vocab.save(VOCAB_DIR+VOCAB_PREFIX+filename[-4:])

    # create phrases (aka bigrams) in the abstract data
    phrases_series = PrepareInput.from_series(sentences_series, vocab)

    arxiv_processed = pd.concat([phrases_series, msc_series], axis=1)

    # drop rows with no valid labels (MSC codes)
    no_valid_label = arxiv_processed.loc[msc_series.apply(len)==0].index
    arxiv_processed = arxiv_processed.drop(no_valid_label)

    # drop rows with < MIN_ABSTR words in the abstract
    too_short = arxiv_processed.loc[phrases_series.apply(len)<MIN_ABSTR].index
    arxiv_processed = arxiv_processed.drop(too_short)

    proc_path = ARXIV_PROC_DIR + PROCESSED_PREFIX + filename
    pd.to_pickle(arxiv_processed, proc_path)
