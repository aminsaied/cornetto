#!/usr/bin/env python3
"""Multiplies arxiv data with multiple labels.

Replaces one paper in the arxiv with N different class labels, with N papers
each with one label. Most papers have multiple labels, so this allows the
model to have a large training data set.

Note: this script might take a while to run.
"""
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.containers import MSC
from modules.arxiv_processor import PrepareOutput
from modules.datasets import load_arxiv, search_files

DEPTH = 2
POSSIBLE_CODE_LENGTHS = ['2','3','5']
ARXIV_PROC_DIR = '../data/arxiv/'
REPEATED_PREFIX = '-repeated_'

if __name__ == "__main__":

    depth_str = input("Input depth of the MCS codes \
                        to compute TF-IDF for. Pick 2,3 or 5\n")
    if depth_str not in POSSIBLE_CODE_LENGTHS:
        depth_str = DEPTH
        print("Was chosen the default value = %s"%DEPTH)
    depth = int(depth_str)
    msc_bank = MSC.load(depth)

    print("Loading a processed arxiv dataset...")
    print("Choose arxiv file.")
    arxiv_file = search_files(ARXIV_PROC_DIR)
    arxiv_path_name = ARXIV_PROC_DIR+arxiv_file
    arxiv = load_arxiv(path_and_filename=arxiv_path_name, depth=depth)

    print("Processing data...")
    print("Processed of ouf %s entries: "%(len(arxiv.index)))
    arxiv_repeated = PrepareOutput.from_dataframe(arxiv,msc_bank=msc_bank)
    print("Done.\n Saving the results...")
    pd.to_pickle(arxiv_repeated, ARXIV_PROC_DIR+REPEATED_PREFIX+arxiv_file)
    print("Done.")
