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
from modules.containers import Vocab
from modules.datasets import load_arxiv, search_files

VOCAB_DIR = '../data/vocab/' 
ARXIV_PROC_DIR = '../data/arxiv/'

if __name__ == "__main__":

    print('Choose a vocabulary file to use.')
    vocab_file = search_files(VOCAB_DIR)
    print("Vocab file: ", vocab_file)
    # need to use '-4' to remove the extension...
    # TODO: re-write it pretty
    vocab = Vocab.load(VOCAB_DIR+vocab_file[:-4])
    
    print("Loading the arxiv dataset...")
    print("Choose arxiv file to train TF-IDF on.")
    print("Note: each text should have one label (e.g., use 'repeated' arxiv)")
    arxiv_file = search_files(ARXIV_PROC_DIR)
    arxiv_path_name = ARXIV_PROC_DIR+arxiv_file
    arxiv = load_arxiv(path_and_filename=arxiv_path_name)
    
    w2v_model = GensimWordToVec()
    print("Training the Word-to-Vec model.")
    w2v_model.fit(arxiv['Abstract'],vocab)
    W2V_FILE = 'w2v_model'
    w2v_model.save(W2V_FILE)
    print("The model is saved by the name {}".format(w2v_model.PATH+W2V_FILE))
    
    