#!/usr/bin/env python3
"""Script computing TF-IDF scores for a given vocabulary.

Incorporates feature reduction by dropping the words with the lowest TF-IDF
score. It uses a pre-processed arxiv dataset to learn the TF-IDF scores, where
each text (abstract) corresponds to only one label.

Such a pre-processed dataset is constructed by running 'repeat_arxiv_data.py'.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.container_builders import TFIDFBuilder
from modules.containers import MSC, Vocab
from modules.datasets import load_arxiv, search_files

DEFAULT_DEPTH = 2
POSSIBLE_CODE_LENGTHS = ['2','3','5']
VOCAB_DIR = '../data/vocab/'
REDUCED_PREFIX = '-reduced_'
ARXIV_PROC_DIR = '../data/arxiv/'
TFIDF_PREFIX = '-tfidf_'

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
    vocab = Vocab.load(VOCAB_DIR+vocab_file[:-4])

    print("Loading the arxiv dataset...")
    print("Choose arxiv file to train TF-IDF on.")
    print("Note: each text should have one label (e.g., use 'repeated' arxiv)")
    arxiv_file = search_files(ARXIV_PROC_DIR)
    arxiv_path_name = ARXIV_PROC_DIR+arxiv_file
    arxiv = load_arxiv(path_and_filename=arxiv_path_name, depth=depth)

    ind = lambda code: msc_bank[code].id
    labels = arxiv['MSCs'].apply(ind).tolist()
    docs = arxiv['Abstract'].tolist()

    print("Computing TFIDF scores. This may take a while...")
    tfidf_builder = TFIDFBuilder()
    tfidf = tfidf_builder.build(vocab, docs, labels)
    print("Saving...")
    tfidf.save(VOCAB_DIR+TFIDF_PREFIX+vocab_file)
    print("Done.")

    print("The current size of vocabulary is %s"%(len(vocab)))
    try:
        size = int( input("How many words would you like to keep?\n") )
    except (ValueError, TypeError):
        size = len(vocab)

    print("Reducing the size of the vocabulary...")
    keys_to_keep = tfidf.get_N_highest(size,attr='tfidf')
    new_vocab = vocab.keep_keys(keys_to_keep)

    vocab_new = VOCAB_DIR+REDUCED_PREFIX+vocab_file
    new_vocab.save(vocab_new)
    print("The new vocabulary is saved by the name:\n %s "%vocab_new)
