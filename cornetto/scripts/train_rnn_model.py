"""
This script trains a Gensim Word-to-Vec model 
from a pre-processed arxiv dataset. 

Such a pre-processed dataset is constructed 
by running the 'repeat_arxiv_data.py' script.

This w2v model is used to create and train RNN model
math text classification.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.word_to_vec_builders import GensimWordToVec
from modules.datasets import load_arxiv, search_files
from modules.data_handlers import RNNTrainingData
from modules.containers import Vocab, MSC
from modules.rnn_model import RNNModel

VOCAB_DIR = '../data/vocab/' 
ARXIV_PROC_DIR = '../data/arxiv/'
DEFAULT_DEPTH = 2
POSSIBLE_CODE_LENGTHS = ['2','3','5']

if __name__ == "__main__":
    depth_str = input("Input depth of the MCS codes \
                        to compute TF-IDF for. Pick 2,3 or 5\n")
    if depth_str not in POSSIBLE_CODE_LENGTHS:
        depth_str = DEFAULT_DEPTH
        print("Was chosen the default value = %s"%DEFAULT_DEPTH)
    depth = int(depth_str)
    msc_bank = MSC.load(depth)
    
    print('Choose a vocabulary file to use.')
    vocab_file = search_files(VOCAB_DIR)
    # need to use '-4' to remove the extension...
    # TODO: re-write it pretty
    vocab = Vocab.load(VOCAB_DIR+vocab_file[:-4])
    
    print("Loading the arxiv dataset...")
    print("Choose arxiv file to train RNN on.")
    arxiv_file = search_files(ARXIV_PROC_DIR)
    arxiv_path_name = ARXIV_PROC_DIR+arxiv_file
    arxiv = load_arxiv(path_and_filename=arxiv_path_name, depth=depth)
    
    w2v = GensimWordToVec.load('w2v_model')
    model = RNNModel(n_inputs=w2v.dim, n_outputs=len(msc_bank), w2v_model=w2v)
    
    print("Creating training data..")
    print("This may take some time...")
    training_data = RNNTrainingData.build(arxiv, w2v, msc_bank)
    print("Done. Training the model...")
    model.fit(training_data)
    filename = input("Enter the file to save the model")
    model.save(filename)
    print("The model was saved.")