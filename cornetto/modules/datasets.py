import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from arxiv_processor import MSCCleaner
from containers import MSC
from data_handlers import UnigramTrainingData
from data_handlers import RNNTrainingData
from data_handlers import WordToVecModel

ROOT_DIR = '../data/training_data/'
UNIGRAM_DIR = ROOT_DIR + '-unigram_training_data_DEPTH'
RNN_DIR = ROOT_DIR + "-rnn_training_data_DEPTH"

def load_arxiv(depth = 5):
    """
    Loads processed arxiv of specified depth.
    See 'containers' for details of processing.
    -- depth: int (default = 5)
    """
    DIR = '../data/arxiv/'
    ARXIV = "-arxiv_processed"
    EXT = ".pkl"

    arxiv = pd.read_pickle(DIR+ARXIV+EXT)

    if depth != 5:
        arxiv['MSCs'] = MSCCleaner._specify(arxiv['MSCs'], depth=depth)

    return arxiv

def load_word2vec(dim=50):
    """
    Loads standard word2vec model.
    -- dim: int default(50), 50, 70 or 100, dimension of the word vectors
    """
    DIR = './data/word2vec_models/'
    W2V_MODEL = '-w2v_model_'
    return WordToVecModel.load(DIR+W2V_MODEL+str(dim))

def load_rnn_training_data_builder(selection = None, depth = 5):
    """Loads standard RNN training data builder."""
    w2v_model = load_word2vec()
    msc_bank = MSC.load(depth)
    rnn_td_builder = RNNTrainingData(w2v_model, msc_bank, selection=selection)
    return rnn_td_builder

def training_data(kind, depth = 5):
    """
    Given a type of training data, returns training data object of that kind.
    kind: string, e.g. 'unigram'
    depth: int, either 2, 3 or 5(default)
    """

    if kind == 'unigram':
        return UnigramTrainingData.load(UNIGRAM_DIR + str(depth))

    if kind == 'rnn':
        return RNNTrainingData.load(RNN_DIR + str(depth))

def demonstration_examples(kind):
    """
    Loads example data designed for demonstration scripts.
    """

    DIR = './data/demos/'
    RNN_DEMO = "-demo_rnn_examples"
    EXT = '.pkl'

    if kind == 'rnn':
        return pd.read_pickle(DIR+RNN_DEMO+EXT)
