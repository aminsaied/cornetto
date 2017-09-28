# uses Python 3
import pandas as pd
import numpy as np
from itertools import combinations

from text_processor import WordSelector

class PairedData(object):
    """
    Object to handle labeled data of the form (x,y).
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
        return TrainingData(training, self.test, self.validation)

    @classmethod
    def build(cls):
        pass

    def save(self, filename):
        EXTENSION = '.pkl'
        data = [self.training, self.test, self.validation]
        pd.to_pickle(data, filename+EXTENSION)

    @staticmethod
    def load(filename):
        EXTENSION = '.pkl'
        try:
            data = pd.read_pickle(filename+EXTENSION)
        except FileNotFoundError:
            print("No file by that name. Remember, we add the .pkl extension for you!")
        return TrainingData(*data)

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

class Word2VecTrainingData(T2VTrainingData):

    def __init__(self, X, Y):
        super().__init__(X,Y)

    @classmethod
    def from_dataframe(cls, texts, vocab):
        """
        Given a series of texts, create the training data for Word2Vec model
        """
        word_selector = WordSelector(vocab)
        select = lambda text: word_selector.select_words(text)
        texts_as_words = texts.apply(select)
        temp1, temp2 = zip(*texts_as_words.map(Word2VecTrainingData._create_pairs))
        X = np.hstack(temp1)
        Y = np.hstack(temp2)
        return cls(X,Y)

    @staticmethod
    def _create_pairs(words):
        X, Y = [], []
        PAIR = 2
        if words and (len(words) > 1):
            pairs = combinations(words,r=PAIR)
            X,Y = list(zip(*pairs))
        return np.array(X).reshape(1, len(X)), np.array(Y).reshape(1,len(Y))

class WordToVecModel(object):
    """
    Handles the word2vec dictionary.
    """
    def __init__(self, w2v_dict):
        self._w2v_dict = w2v_dict
        self.dim = self._get_dim()

    def __getitem__(self, word):
        return self._w2v_dict[word]

    def __len__(self):
        return len(self._w2v_dict)

    def __contains__(self, word):
        return (word in self._w2v_dict)

    def _get_dim(self):
        a_vec = list(self._w2v_dict.values())[0]
        return len(a_vec)

    def save(self, filename):
        EXTENSION = ".pkl"
        data = self._w2v_dict
        pd.to_pickle(data, filename+EXTENSION)

    @classmethod
    def load(cls, filename):
        EXTENSION = ".pkl"
        data = pd.read_pickle(filename+EXTENSION)
        return cls(data)

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

    def __init__(self, w2v_model, msc_bank, selection=None, n_steps=50):
        """
        -- w2v_model: WordToVecModel
        -- msc_bank: MSC, see 'containers'
        -- selection: list or (default) None, of codes to keep
        -- test_size: int (default 1000), number of test examples
        -- n_steps: int (default 50) or None, the rnn will only
             read up to this many words. If None, then automatically
             selects length of longest abstract.
        """
        self.n_steps   = n_steps
        self.w2v_model = w2v_model
        self.n_inputs  = w2v_model.dim
        self.msc_bank  = msc_bank
        if selection:
            self.msc_bank = msc_bank.keep_keys(selection)

    def build(self, arxiv):
        """
        Returns a TrainingData object suited to train our RNNs.
        Option to choose a smaller selection of codes on which to train.
        -- arxiv: dataframe, prepared arxiv dataframe, see 'datasets'
        """
        if self.n_steps == None:
            self.n_steps = np.max(arxiv.Abstract.apply(len).tolist())

        arxiv_primary, arxiv_sentences = self._make_input_output(arxiv)

        input_  = self._build_input(arxiv_sentences)
        length_ = self._build_length(arxiv_sentences)
        output_ = self._build_output(arxiv_primary)

        training_data = self._training_data(input_, length_, output_)

        return training_data

    def _make_input_output(self, arxiv, primary=True):
        """
        Returns input/output series containing only those papers
        with the desired msc codes.
        -- arxiv: DataFrame, processed dataframe, see 'datasets'
        """
        arxiv_shuffled = arxiv.sample(frac=1, random_state=73).reset_index(drop=True)

        in_selection = lambda codes: (codes[0] in self.msc_bank)
        valid = arxiv_shuffled.MSCs.apply(in_selection)
        arxiv_sample = arxiv_shuffled[valid != False].reset_index(drop=True)
        arxiv_sentences = arxiv_sample.Abstract

        if primary:
            arxiv_codes = self._select_primary(arxiv_sample.MSCs)
        else:
            arxiv_codes = self._select_all_codes(arxiv_sample.MSCs, self.msc_bank)

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

    def _build_input(self, arxiv_sentences):
        """
        Returns array of np arrays built from sentences by
        containing the corresponding word vectors (padded
        with zero vectors).
        -- arxiv_sentences: series
        """
        f = lambda sentence: self._build_rnn_input(sentence)
        input_series = arxiv_sentences.apply(f)
        return np.array(input_series.tolist())

    def _build_output(self, arxiv_primary):
        """
        Returns array of primary MSC codes' indices. Looks up
        index of MSC code in the msc_bank.
        -- arxiv_primary: series, of codes (as strings)
        """
        f = lambda code: self.msc_bank[code].id
        output_series = arxiv_primary.apply(f)
        return np.array(output_series.tolist())

    def _build_output_with_all_codes(self, arxiv_primary):
        """
        Returns array of all MSC codes'.
        -- arxiv_primary: series, of codes (as strings)
        """
        one_hot_codes = arxiv_primary.apply(self.msc_bank.one_hot).reset_index(drop=True)
        convert_to_int = lambda index: [int(id_) for id_ in index]
        y = one_hot_codes.apply(convert_to_int)
        return np.array(y.tolist())

    def _build_length(self, arxiv_sentences):
        return np.array(arxiv_sentences.apply(len).tolist())

    def _build_rnn_input(self, sentence):
        """
        Returns a 2-tensor of shape=(n_steps, n_inputs),
        i.e., with rows = word_vectors
        Fills the remaining rows with zeros
        """
        sent = sentence[:self.n_steps] # trim sentence
        vecs = np.array([self.w2v_model[w] for w in sent if w in self.w2v_model])
        input_ = np.zeros((self.n_steps, self.n_inputs))
        input_[:vecs.shape[0]] = vecs
        return input_

    def _training_data(self, input_, length_, output_, test_cut_off=1000):
        """
        Returns a TrainingData object holding the training and test data for an RNN.
        TrainingData consists of RNNData, which itself is a triple of (X, length, Y).
        """
        test_size = min(len(output_)*10//100, test_cut_off)
        dim_output = len(self.msc_bank)

        input_train, input_test   = input_[:-test_size], input_[-test_size:]
        length_train, length_test = length_[:-test_size], length_[-test_size:]
        output_train, output_test = output_[:- test_size], output_[-test_size:]

        training = RNNData(input_train, length_train, output_train, dim_output)
        testing  = RNNData(input_test, length_test, output_test, dim_output)

        return TrainingData(training, testing)

    def training_data_with_all_codes(self, arxiv):
        """Keeps all (not only primary) MSC codes."""

        if self.n_steps == None:
            self.n_steps = np.max(arxiv.Abstract.apply(len).tolist())

        y, arxiv_sentences = self._make_input_output(arxiv, primary=False)

        input_  = self._build_input(arxiv_sentences)
        length_ = self._build_length(arxiv_sentences)

        return input_, length_, y
