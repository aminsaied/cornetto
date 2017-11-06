import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from arxiv_processor import MSCCleaner
from containers import MSC
from data_handlers import UnigramTrainingData
from data_handlers import RNNTrainingData

ROOT_DIR = '../data/training_data/'
UNIGRAM_DIR = ROOT_DIR + '-unigram_training_data_DEPTH'
RNN_DIR = ROOT_DIR + "-rnn_training_data_DEPTH"

INPUT_ = 'Abstract'
OUTPUT_ = 'MSCs'

def search_files(data_dir):
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        print( 'Which file would you like to select?')
        for i, filename in enumerate(filenames):
            print('%s. %s'%(i, filename))
        index_str = input('Make a selection. Empty input to quit.\n')
        try:
            index = int(index_str)
        except (ValueError, TypeError):
            return
        assert index in range(len(filenames))
        return filenames[index]

def read_raw_arxiv_data(path_and_filename):
    """
    NOTE: filename should have extention!
    """
    print("Reading data...")
    try:
        df = pd.read_pickle(path_and_filename)
    except FileNotFoundError:        
        print("File not found.")
        df = pd.DataFrame(columns=[INPUT_, OUTPUT_])
    
    return df[[INPUT_, OUTPUT_]]
        
def load_arxiv(path_and_filename=None, depth = 5):
    """
    Loads *processed* arxiv of specified depth.
    See 'containers' for details of processing.
    -- depth: int (default = 5)
    """
    if not path_and_filename:
        path_and_filename = '../data/arxiv/-arxiv_processed.pkl'

    arxiv = pd.read_pickle(path_and_filename)

    if depth != 5:
        arxiv['MSCs'] = MSCCleaner.specify_depth(arxiv['MSCs'], depth=depth)

    return arxiv

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
