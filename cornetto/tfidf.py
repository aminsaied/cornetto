import numpy as np
import pandas as pd

from container_builders import TFIDFBuilder
from containers import MSC, Vocab
from arxiv_processor import PrepareOutput
import datasets

if __name__ == "__main__":

    DEPTH = 2
    VOCAB_PREV = '-vocab'
    VOCAB_NEW = '-vocab_reduced'

    msc_bank = MSC.load(DEPTH)
    vocab = Vocab.load(VOCAB_PREV)
    print("Loading the arxiv dataset...")
    arxiv = datasets.load_arxiv(depth=2)
    print("Preparing the TFIDF training data...")
    print("Processed of ouf %s entries: "%(len(arxiv)))
    arxiv_repeated = PrepareOutput.from_dataframe(arxiv,msc_bank=msc_bank)
    pd.to_pickle(arxiv_repeated, '-arxiv_repeated.pkl')
    
    ind = lambda code: msc_bank[code].id
    labels = arxiv_repeated['MSCs'].apply(ind).tolist()
    docs = arxiv_repeated['Abstract'].tolist()
    
    print("Computing TFIDF scores. This may take a while...")
    tfidf_builder = TFIDFBuilder()
    tfidf = tfidf_builder.build(vocab, docs, labels)
    tfidf.save('tfidf_orig_vocab')
    print("Done.")
    
    print("The current size of vocabulary is %s"%(len(vocab)))
    print("How many words would you like to keep?")
    size = int( input() )
    
    print("Reducing the size of the vocabulary...")
    keys_to_keep = tfidf.get_N_highest(size,attr='tfidf')
    new_vocab = vocab.keep_keys(keys_to_keep)
    
    new_vocab.save(VOCAB_NEW)
    print("The new vocabulary is saved by the name  "+VOCAB_NEW)