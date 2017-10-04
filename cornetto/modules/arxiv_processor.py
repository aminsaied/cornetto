# import standard libraries
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from containers import MSC

class MSCCleaner(BaseEstimator, TransformerMixin):
    """
    Given a pandas series of MSC strings, returns a list of (unique) MSC codes,
    of specified depths.
    NB. depth = 2, 3, or 5 (default)
    """

    DEPTH = 5
    def __init__(self, depth=DEPTH):
        self.depth         = depth
        self.msc_bank     = MSC.load(depth)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_as_list = self._to_list(X)
        X_specified = self._specify(X_as_list, self.depth)
        X_valid = self._are_valid(X_specified, self.msc_bank)

        return X_valid

    @staticmethod
    def _to_list(X):
        """
        Converts string of MSCs to list of MSCs
        """
        to_list = lambda codes_string: codes_string.split()
        return X.apply(to_list)

    @staticmethod
    def _specify(X, depth):
        """
        Selects MSC codes of desired depth 2, 3 or 5 digit. Removes
        duplicate codes that this may create.
        """
        specify = lambda codes: [code[:depth] for code in codes]
        MSCs_contracted = X.apply(specify)
        remove_duplicates = lambda codes: list(set(codes))
        return MSCs_contracted.apply(remove_duplicates)

    @staticmethod
    def _are_valid(X, msc_bank):
        """
        Checks MSC codes are valid.
        """
        validate = lambda codes: [code for code in codes if code in msc_bank]
        return X.apply(validate)

class PrepareInput(object):
    """
    This class is a collection of methods, whose purpose is 
    to take a dataframe or a series containing lists of strings 
    (lists of words, a.k.a. sentences) and only keep the ones appearing 
    in a given vocabulary. This includes checking pairs of words 'w1_w2'. 
    We call such pairs 'phrases'.
    """
    @classmethod
    def from_dataframe(cls, df, vocab):
        """
        Given a dataframe with two columns, 
        returns a new dataframe with two columns, input and output.
        Input is a list of words 
        (and new words of the form 'w1_w2' for words w1,w2. 
        Such words we call 'phrases') in vocab. 
        We refer to lists of words in vocab as 'sentences'
        Input:
          -- df: pd.DataFrame, with two columns, the first column 
          contains list of words (strings)
          -- vocab: Vocab, of words and phrases
        Output:
          -- pd.Dataframe, the new dataframe
        """
        input_, output_ = df.columns.values
        f = lambda sentence: cls.from_sentence(sentence, vocab)
        processed_col =  df[input_].apply(f)
        new_df = pd.concat([processed_col, df[output_]], axis=1)
        return new_df

    @classmethod
    def from_series(cls, series, vocab):
        """
        # Given a series, each entry being a list of words (sentence),
        returns a new pd.Series with each antry a list of words 
        (including new words of the form 'w1_w2' 
        for words w1,w2 from the original sentence. 
        Such new words we call 'phrases') in vocab. 
        Input:
          -- series: pd.Series, each entry a list of words (strings)
          -- vocab: Vocab, of words and phrases
        Output:
          -- pd.Series, the new series
        """
        f = lambda sentence: cls.from_sentence(sentence, vocab)
        new_series = series.apply(f)
        return new_series

    @staticmethod
    def from_sentence(sentence, vocab):
        """
        Given a list of words (sentence), returns a list of 
        words in it that appear in vocab. Also checks appearanec of words 
        of the form 'w1_w2' for words w1,w2 from the original sentence.
        Input:
          -- sentence : List[str], list of words
          -- vocab : Vocab, vocabulary
        Output:
          -- List[str]
        """
        words_in_vocab = []
        for w1, w2 in zip(sentence, sentence[1:]):
            if w1 in vocab:
                words_in_vocab.append(w1)
            phrase = w1+'_'+w2
            if phrase in vocab:
                words_in_vocab.append(phrase)
        return words_in_vocab
        
        return cls._in_vocab(sentence, vocab)

class PrepareOutput(object):
    """
    This class is designed to transform a dataframe with two columns of 
    inputs-outputs with several possible outputs for a given input 
    into a gataframe with one input/ one output by repeating the inputs. 
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
        
    """
    def from_dataframe(df, msc_bank=MSC.load(5)):
        """
        Transforms a dataframe with two columns, inputs and outputs, 
        with several possible outputs for a given input, 
        into a gataframe with one input/ one output by repeating the inputs. 
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
        Only uses codes from the msc_bank as potential outputs. 
        Note: if msc_bank containes codes of length < 5, 
        it will strip the codes of the correct length from the outputs.
        I.e., output '14B13' will become '14' if msc_bank 
        has codes of length 2.
        
        Input:
          -- df : pd.Dataframe
          -- msc_bank : MSC object. Defaults to 
          the MSC codes with 5 characters in them.
        Output:
          -- pd.DataFrame, the new dataframe
        """
        input_, output_ = df.columns
        df_proc = pd.DataFrame(columns = [input_, output_])
        count = 0
        # code length will be the same among all the keys, so use the first
        code_length = len(msc_bank.sorted_keys[0])
        for index, row in df.iterrows():
            # printing the progress
            if index % 5000 == 0:
                print(index)
            
            get_prefix = lambda code: code[:code_length]
            prefixes = list(map(get_prefix,row[output_] ))
            valid_codes = [prefix for prefix in prefixes if prefix in msc_bank]
            # repeat the input for different outputs, add as new row
            for code in list(set(valid_codes)):
                new_row = [row[input_], code]
                df_proc.loc[count] = new_row
                count += 1
        return df_proc