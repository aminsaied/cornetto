"""
*NOTE*: this script might take a while to run.
This script transforms a dataframe with two columns, 
inputs and outputs, with several possible outputs for a given input, 
into a dataframe with one input/one output by repeating the inputs. 
I.e., for example dataframe 
    ______________________
    'input_1' | ['a','b']
    'input_2' | ['c']
    __________|___________
will be transformed into
    _________________
    'input_1' | 'a'
    'input_1' | 'b'
    'input_2' | 'c'
    __________|______
Only uses valid MSC codes of a given depth (2,3 or 5) 
as potential outputs. 
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