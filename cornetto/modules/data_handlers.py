#!/usr/bin/env python3
"""Manages training data for the various available models.

Training, development and test data are used throughout this library in
various machine learning algorithms: naive Bayes, text-to-vec and RNN models.
Here we provide classes specifically designed to interface with those models.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from functools import partial

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

class PairedData(object):
    """
    Object to handle labeled data of the form (X,Y).
    """
    def __init__(self, X, Y):
        """
        X, Y are 2D numpy arrays, with matching number of columns
        """
        self.X = X
        self.Y = Y

        self.dim_in = X.shape[0]
        self.dim_out = Y.shape[0]

    def __getitem__(self, key):
        return PairedData(self.X[key], self.Y[key])


class TrainingData(object):
    """
    Supervised machine learning models require training data, test data,
    and frequently cross validation data sets.
    """
    def __init__(self, training_data, test_data, validation_data = None):
        """
        -- training_data: PairedData
        -- validation_data: PairedData, optional
        -- test_data: PairedData
        """
        self.training = training_data
        self.validation = validation_data
        self.test = test_data

    def __getitem__(self, slice):
        """
        Facilitates standard numpy slicing on training data.
        Note: slices only the training data.
        """
        numpy_slice = np.s_[:, slice]
        training = self.training[numpy_slice]
        return self.__class__(training, self.test, self.validation)

    @classmethod
    def build(cls):
        pass

    def save(self, filename):
        EXTENSION = '.pkl'
        data = [self.training, self.test, self.validation]
        pd.to_pickle(data, filename+EXTENSION)

    @classmethod
    def load(cls, filename):
        EXTENSION = '.pkl'
        try:
            data = pd.read_pickle(filename+EXTENSION)
        except FileNotFoundError:
            print("No file %s. Remember, we add the .pkl extension for you!"%filename)
        return cls(*data)


class UnigramTrainingData(TrainingData):
    """Creates and handles unigram model training data."""

    @classmethod
    def build(cls, X, Y, vocab, msc_bank):
        """
        Creates two PairedData objects, training and test.
        -- X: series, abstracts as sentences
        -- Y: series, msc codes as list
        -- vocab: Vocab, see 'containers'
        -- msc_bank: MSC, see 'containers'

        NOTE: training and test differ in type!
        -- self.training.X: list, of selected words
        -- self.training.Y: list, of selected words
        -- self.test.X: array, columns are word vectors of size len(vocab)
        -- self.test.Y: list, columns are msc vectors of size len(msc_bank)
        """

        X_training, test_abstracts = cls._make_cut(X)
        Y_training, test_mscs = cls._make_cut(Y)

        training_data = PairedData(X_training, Y_training)

        X_test, Y_test = cls._build_test_matrices(test_abstracts, test_mscs, vocab, msc_bank)
        test_data = PairedData(X_test, Y_test)

        return cls(training_data = training_data, test_data = test_data)

    @staticmethod
    def _build_test_matrices(test_abstracts, test_mscs, vocab, msc_bank):
        """Returns matrices X and Y for testing model"""
        assert len(test_abstracts) == len(test_mscs)
        N = len(test_abstracts)
        X_test = np.zeros( (len(vocab), len(test_abstracts)) )
        Y_test = np.zeros( (len(msc_bank), len(test_mscs)) )
        abstr_msc = list(zip(test_abstracts, test_mscs))

        for data_ind in range(N):
            abstract, list_of_codes = abstr_msc[data_ind]
            for code in list_of_codes:
                msc_ind = msc_bank[code].id
                for word in abstract:
                    vocab_ind = vocab[word].id
                    X_test[vocab_ind][data_ind] += 1
                Y_test[msc_ind][data_ind] += 1
        Y_test /= np.sum(Y_test, axis=0)
        return X_test, Y_test

    @staticmethod
    def _make_cut(series):
        """Cuts a series in two pieces"""
        CUT_PERCENTAGE = 0.8
        cut = int(len(series)*CUT_PERCENTAGE)
        return series[:cut], series[cut:]

    @staticmethod
    def _get_names(df):
        names = df.columns.values
        NUMBER_OF_COLS = 2
        assert names.size == NUMBER_OF_COLS
        return names[0], names[1]


class T2VTrainingData(TrainingData):

    @classmethod
    def build(cls, df, msc_bank, t2v_model):
        return cls(*cls._from_dataframe(df, msc_bank, t2v_model))

    @staticmethod
    def _from_dataframe(arxiv_processed, msc_bank, t2v_model):
        names = arxiv_processed.columns.values
        NUMBER_OF_COLS = 2
        assert names.size == NUMBER_OF_COLS
        input_, output_ = names[0], names[1]

        df_processed = pd.DataFrame(columns = names)
        df_processed[input_] = T2VTrainingData._build_input_vector(arxiv_processed[input_], t2v_model)
        df_processed[output_] = T2VTrainingData._build_output_vector(arxiv_processed['Code'], msc_bank)

        df_processed = df_processed[df_processed[input_].map(len)>0]
        df_processed = df_processed[df_processed[output_].map(len)>0]

        X = np.hstack(df_processed[input_].tolist())
        Y = np.hstack(df_processed[output_].tolist())

        where_to_cut = T2VTrainingData._where_to_cut(len(arxiv_processed))
        return T2VTrainingData._cut(X,Y,where_to_cut)

    @staticmethod
    def _build_input_vector(series, t2v_model):
        return series.apply(t2v_model.convert)

    @staticmethod
    def _build_output_vector(series, msc_bank):
        return series.apply(msc_bank.one_hot)

    @staticmethod
    def _where_to_cut(n_samples):
        FIRST_CUT_PERCENT = 80
        SECOND_CUT_PERCENT = 90
        TOTAL = 100
        return n_samples*FIRST_CUT_PERCENT//TOTAL, n_samples*SECOND_CUT_PERCENT//TOTAL

    @staticmethod
    def _cut(X, Y, cuts):
        print("Shape of X", X.shape)
        print("Shape of Y", Y.shape)
        training = PairedData( X[:, : cuts[0] ], Y[:, : cuts[0] ] )
        test = PairedData( X[:, cuts[0] : cuts[1] ], Y[:, cuts[0] : cuts[1] ] )
        validation = PairedData( X[:, cuts[1] : ], Y[:, cuts[1] : ] )
        return training, test, validation


class RNNData(object):
    """
    Object to handle rnn training data of the form (X, length, Y, dimY)
    where:
    -- X: 3D - array, shape (n_examples, n_words, dim_word_vectors)
    -- length: 1D array, lengths of abstracts
    -- Y: 1D array, indices of MSC codes
    -- dimY: int (default None), number of possible MSC codes
    """
    def __init__(self, X, length, Y, dimY=None):
        self.X = X
        self.length = length
        self.Y = Y
        self.dimY = dimY


class RNNTrainingData(TrainingData):
    """
    Creates and handles training data for our recurrent neural networks.
    """

    @classmethod
    def build(cls, arxiv, w2v_model, msc_bank, n_steps=50):
        """
        Returns a TrainingData object suited to train our RNNs.
        Option to choose a smaller selection of codes on which to train.
        -- arxiv: dataframe, prepared arxiv dataframe, see 'datasets'
        """
        # TODO: make sure that both primary and non=primary work fine

        arxiv_codes, arxiv_sentences = cls._make_input_output(arxiv, msc_bank)

        input_  = cls.build_input(arxiv_sentences, w2v_model, n_steps)
        length_ = cls._build_length(arxiv_sentences)
        output_ = cls.build_output(arxiv_codes, msc_bank)

        dim_output = len(msc_bank)
        return cls(*cls._make_cut(input_, length_, output_, dim_output))

    @classmethod
    def _make_input_output(cls, arxiv, msc_bank, primary=True):
        """
        Returns input/output series containing only those papers
        with the desired msc codes.
        -- arxiv: DataFrame, processed dataframe, see 'datasets'
        """
        arxiv_shuffled = arxiv.sample(frac=1, random_state=73).reset_index(drop=True)

        in_selection = lambda codes: (codes[0] in msc_bank)
        valid = arxiv_shuffled.MSCs.apply(in_selection)
        arxiv_sample = arxiv_shuffled[valid != False].reset_index(drop=True)
        # TODO: do not use the names of the columns!!!
        arxiv_sentences = arxiv_sample.Abstract

        if primary:
            arxiv_codes = cls._select_primary(arxiv_sample.MSCs)
        else:
            arxiv_codes = cls._select_all_codes(arxiv_sample.MSCs, msc_bank)

        return arxiv_codes, arxiv_sentences

    @staticmethod
    def _select_primary(msc_series):
        """Returns series with index of primary msc code."""
        primary_code = lambda codes: codes[0]
        arxiv_primary = msc_series.apply(primary_code)
        return arxiv_primary

    @staticmethod
    def _select_all_codes(msc_series, msc_bank):
        """Returns one-hot array of msc_codes."""
        one_hot_codes = msc_series.apply(msc_bank.one_hot).reset_index(drop=True)
        convert_to_int = lambda index: [int(id_) for id_ in index]
        y = one_hot_codes.apply(convert_to_int)
        return np.array(y.tolist())

    @classmethod
    def build_input(cls, arxiv_sentences, w2v_model, n_steps):
        """
        Returns array of np arrays built from sentences by
        concatenating the corresponding word vectors (padded
        with zero vectors).
        -- arxiv_sentences: series
        """
        f = partial(cls.build_rnn_input, w2v_model=w2v_model, n_steps=n_steps )
        input_series = arxiv_sentences.apply(f)
        return np.array(input_series.tolist())

    @staticmethod
    def build_output(arxiv_primary, msc_bank):
        """
        Returns array of primary MSC codes' indices. Looks up
        index of MSC code in the msc_bank.
        -- arxiv_primary: series, of codes (as strings)
        """
        f = lambda code: msc_bank[code].id
        output_series = arxiv_primary.apply(f)
        return np.array(output_series.tolist())

    @staticmethod
    def _build_length(arxiv_sentences):
        return np.array(arxiv_sentences.apply(len).tolist())

    @staticmethod
    def build_rnn_input(sentence, w2v_model, n_steps):
        """
        Returns a 2-tensor of shape=(n_steps, n_inputs),
        i.e., with rows = word_vectors
        Fills the remaining rows with zeros
        """
        sent = sentence[:n_steps] # trim sentence
        vecs = np.array([w2v_model[w] for w in sent if w in w2v_model])
        input_ = np.zeros((n_steps, w2v_model.dim))
        input_[:vecs.shape[0]] = vecs
        return input_

    @staticmethod
    def _make_cut(input_, length_, output_, dim_output, test_cut_off=1000):
        """
        Returns a TrainingData object holding the training and test data for an RNN.
        TrainingData consists of RNNData, which itself is a triple of (X, length, Y).
        """
        test_size = min(len(output_)*10//100, test_cut_off)

        input_train, input_test   = input_[:-test_size], input_[-test_size:]
        length_train, length_test = length_[:-test_size], length_[-test_size:]
        output_train, output_test = output_[:- test_size], output_[-test_size:]

        training = RNNData(input_train, length_train, output_train, dim_output)
        test  = RNNData(input_test, length_test, output_test, dim_output)

        return training, test
