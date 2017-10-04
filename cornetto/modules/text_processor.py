# uses Python 3

# import standard libraries
import numpy as np
import pandas as pd
import re

#import methods from libraries
from nltk.tag import pos_tag
from collections import Counter, namedtuple, UserDict

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from arxiv_processor import PrepareInput

class TextCleaner(object):
    """
    This is a collection of static methods designed to select 
    'appropriate' words from a text.
    """
    @staticmethod
    def strip_math(text):
        """
        Replaces all the LaTeX math in a given text with spaces.
        Input:
          -- text : string
        Output:
          string : text with math removed
        """
        no_dollar_signs = re.sub( r'\$+[^\$]+\$+',r' ', text)
        no_math = re.sub( r'\\\[[^\]]+\]', r' ', no_dollar_signs )
        return no_math

    @staticmethod
    def strip_non_alphas(text):
        """
        Removes non-letter symbols, replacing them with spaces. 
        Note: also removes tabs and new line symbols. 
        Input:
          -- text : string
        Output:
          string : text with all non-letter symbols removed
        """
        no_dash_apos_pattern = re.compile('[-\']+')
        no_dash_apos = re.sub(no_dash_apos_pattern, ' ', text)
        noalpha_pattern = re.compile('[^a-zA-Z\\s]+')
        noalpha = re.sub(noalpha_pattern, '', no_dash_apos)
        paragraph_marker_pattern = re.compile('\n')
        return re.sub(paragraph_marker_pattern, ' ', noalpha)

    @classmethod
    def get_clean_words(cls, text):
        """
        Removes math and non-alpha characters from the text, and
        returns it as a list of words.
        Input:
          -- text : string
        Ouput:
          List[str] : list of words
        """
        clean_text = cls.clean(text)
        return clean_text.split()

    @classmethod
    def clean(cls, text):
        """
        Removes math and non-alpha characters 
        (including tabs and new line symbols) from the text.
        Input:
          -- text : string
        Output:
          string, 'cleaned' text
        """
        no_math = cls.strip_math(text)
        clean_text = cls.strip_non_alphas(no_math)
        return clean_text

class POSTagger(object):
    """
    This class wraps the nltk part-of-speach (POS) tagger.
    """
    @classmethod
    def tag_text(cls, text):
        """
        Given text, removes all math and non-alpha symbols, 
        splits it into words and tags them with POS tag.
        Input:
          -- text, string
        Output:
          List[tup] : list of pairs (word , tag), both being strings, 
            for every word from the text 
            (after it was cleaned by the TextCleaner).
        """
        word_list = TextCleaner.get_clean_words(text)
        tagged_words = cls.tag_words(word_list)
        return tagged_words
        
    @staticmethod
    def tag_words(words):
        """
        Given a list of words, tag them with their part of speech.
        Input:
          -- List[str] : list of words.
        """
        return pos_tag(words)

class WordSelector(object):
    """Converts a text into a list of words of the specified POS."""
    @classmethod
    def __init__(self, vocab):
        """
        Input:
          -- vocab : any object that can test containment of words
        e.g. Vocab object, WordToVecModel object, list of words
        """
        self.vocab = vocab

    def select_N_words(self, text, N_words=25):
        """
        Randomly selects and returns a given number of words from the text.
        Only takes words that are in self.vocab. 
        Note: selection is WITH replacement.
        Input:
          -- text : str 
          -- N_words : int, the number of words to select from text
        Output:
          List[str] : list of selected words (strings)
        """
        all_words = self.select_words(text)
        if all_words:
            words = np.random.choice(all_words,N_words).tolist()
        else:
            words = []
        return words

    def select_words(self, text):
        """
        Select words from text that appear in self.vocab. 
        Note: it also checks phrases: words "w1_w2" where w1 and w2 are 
        in self.vocab.
        Input:
          -- text : string
        Ouput:
          List[str] : list of selected words
        """
        sentence = text.split()
        selected_words =  self.select_from_sentence(sentence)
        return selected_words
    
    def select_from_sentence(self, sentence):
        """
        Given a sentence, pick words and phrases (pairs of words) 
        that appear in the vocabulary self.vocab.
        Input:
          -- List[str] : list of words
        Ouput:
          -- List[str] : list of words from self.vocab, now also 
          includes phrases -- strings of the form "w1_w2" 
          where w1 and w2 are in self.vocab.
        """
        selected_words = PrepareInput.from_sentence(sentence, self.vocab)
        return selected_words