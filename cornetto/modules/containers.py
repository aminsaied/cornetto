# uses Python 3

# import standard libraries
import sys
import numpy as np
import pandas as pd
from scipy import sparse

#import methods from libraries
from collections import namedtuple
from copy import deepcopy

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

thismodule = sys.modules[__name__]

import data
# MSC_DATA_PATH = 'data/msc_classes/'
MSC_DATA_PATH = os.path.dirname(data.__file__)
MSC_FILENAME_TEMPLATE = MSC_DATA_PATH+'/msc_classes/' + '%s_digit'

class _SortedContainer(object):
    """
    Abstract container class to be used as a dictionary. 
    One of the main features: allows fast access to the elements (keys) 
    by id (index) and also fast access to the information 
    (usually named tuple) about a given key. 
    
    Class methods: 
      -- feature_cls_name(cls): *abstract property*, name of 
      the class used to encode the features.
      -- load(cls, file, dtype=None): returns a new instance 
      loaded from a file (can be a file or just a filename with a path).

    
    Object of this class has two fields:
      -- sorted_keys : List, list of keys. NOTE: "Sorted" simply refers to 
      the fact this is a list, so there is an order on the elements.
      -- features: this is dictionary {key:value}, 
      key is an element of the self.sorted_keys, value is a named tuple. 
      WARNING: named tuple is assumed to have two fields: 'key' and 'id' 
      with 'key' matching the key and 'id' a unique int.
      
    Object of this class behaves like a "two-sided" dictionary, 
    has indexing, len, etc. Moreover, it has:
      -- save(filename): saves the object into a file. 
      NOTE: filename has no extension, 
      i.e. 'my_container', NOT 'my_container.txt'
      -- one_hot(keys,as_sparse=False): given a key or a list of keys, 
      return a numpy array (column) if dimension=len(self) with '1' 
      at the positions corresponding to the keys.
      -- sorted_by(self, attribute="id", reverse=True): returns 
      a new instance with the same keys but sorted w.r.t. a given attribute
      of the features. 
      -- remove_keys(self, keys_to_remove): return a new instance, 
      all keys from the list keys_to_remove will be omitted.
      -- keep_keys(self, keys_to_keep): return a new instance, 
      only has keys from the list keys_to_keep.
    """
    @property
    def feature_cls_name(cls):
        raise NotImplementedError

    def __init__(self, sorted_keys = [], features = {}):
        self.sorted_keys = sorted_keys
        self._set_features(features)

    def _set_features(self, features):
        self._features = {}
        for index, key in enumerate(self.sorted_keys):
            if key not in features:
                raise KeyError("Key %s not in features"%key)
                return
            feature = features[key]
            self._features[key] = self._set_feature_id(feature,id=index)

    def _set_feature_id(self, feature, id):
        return feature._replace(id=id)
        
    def __setitem__(self, key, item):
        # assert item isinstance(_Feature)
        # TODO: make a general class called _Feature
        self._features[key] = item

    def __getitem__(self, key):
        if isinstance( key, (int, slice) ):
            return self._get_by_index(key)
        elif isinstance( key, list ):
            return self._get_by_list(key)
        else:
            return self._get_by_key(key)

    def _get_by_index(self, index):
        try:
            return self.sorted_keys[index]
        except IndexError:
            print("Key not found")
            pass

    def _get_by_list(self, index_list):
        try:
            found = [self[index] for index in index_list]
            return found
        except IndexError:
            print("Key not found")
            return []

    def _get_by_key(self, key):
        try:
            return self._features[key]
        except KeyError:
            print("Key not found")
            pass

    def __iter__(self):
        for key in self.sorted_keys:
            yield key

    def __contains__(self, key):
        return (key in self._features)

    def __len__(self):
        return len(self.sorted_keys)

    def __str__(self):
        features_as_str = [str(self[key]) for key in self.sorted_keys]
        return '\n'.join(features_as_str)

    def __add__(self, other):
        new_container = deepcopy(self)
        if self.__class__ != other.__class__:
            raise TypeError("You can only add objects of the same type.")
            return

        for key in other:
            feature = other[key]
            if key not in self:
                new_container._add_new(key, feature)
            else:
                new_container._update_feature(key, feature)

        new_container._sort_and_update_ids()
        return new_container

    def _add_new(self, key, feature):
        self[key] = self._set_feature_id(feature, id=len(self))
        self.sorted_keys.append(key)

    def _update_feature(self, key, feature):
        updated_feature = self._get_updated_feature(key, feature)
        self[key] = updated_feature

    def _get_updated_feature(self, key, other_feature):
        return self[key]

    def _sort_and_update_ids(self):
        for index, key in enumerate(self.sorted_keys):
            self[key] = self._set_feature_id(self[key], id=index)

    def save(self, filename):
        """
        Input:
          -- filename: str, *no extension*, 
          i.e. 'my_file' isnstead of 'my_file.txt'
        """
        EXTENSION = '.csv'
        features_as_tups = list(map(tuple, self._features.values()))
        features_as_df = pd.DataFrame(features_as_tups)
        features_as_df.to_csv(filename+EXTENSION, mode = 'w', index=False)
        
    @classmethod
    def load(cls, file, dtype=None, encoding = 'ISO-8859-1'):
        """
           -- file : either a filename or a file object.
           -- dtype : dictionary of types for each feature attribute
        """
        # if file is just a filename
        if isinstance(file, str):
            EXTENSION = '.csv'
            file += EXTENSION
        try:
            features_as_df = pd.read_csv(file, dtype=dtype, encoding=encoding)
        except (ValueError, EOFError):
            print("Can't read the file.")
            empty = cls()
            return empty
        
        # WARNING: this assumes that the class describing the features is 
        # defined withing this module
        feature_cls = getattr(thismodule, cls.feature_cls_name)
        features_list = [feature_cls(*x) for x in features_as_df.values]
        sorted_keys, features = cls._from_tups(features_list)
        return cls(sorted_keys, features)

    @staticmethod
    def _from_tups(features_list):
        id = lambda feature: feature.id
        sorted_features = sorted( features_list, key=id )
        sorted_keys = [feature.key for feature in sorted_features]
        features = dict(zip(sorted_keys,sorted_features))

        return sorted_keys, features
            
    def one_hot(self, key, as_sparse=False):
        """
        Returns numpy *column* with 1 at the index corresponding to the key.
        
        """
        vec = np.zeros( (len(self),1) )
        if isinstance(key, list):
            for item in key:
                vec += self.one_hot(item)
        elif key in self:
            vec[ self[key].id ] = 1

        if as_sparse == True:
            return sparse.csr_matrix(vec)
        else:
            return vec

    def sorted_by(self, attribute="id", reverse=True):
        """
        The method returns new Vocab object, with same keys
        but now sorted by the given attribute.
        Input:
          -- attribute : str, name of the attribute.
        Output:
          -- new instance of self.__class__
        """
        value = lambda x: getattr(x, attribute)
        resorted_keys = sorted(self.sorted_keys, key=value, reverse=reverse)
        new_instance = self.__class__(resorted_keys, self._features)
        return new_instance

    def remove_keys(self, keys_to_remove):
        """
        Return a new container object, 
        with all the keys from keys_to_remove omitted.
        Input:
          -- keys_to_remove : List, list of keys to omit
        Ouput:
          -- new instance of self.__class__
        """
        new_keys = [key for key in self if key not in keys_to_remove]
        new_instance = self.__class__(new_keys, self._features)
        return new_instance

    def keep_keys(self, keys_to_keep):
        """
        Return a new container object built out of the specified list of keys.
        Input:
          -- keys_to_keep : List, list of keys to keep
        Ouput:
          -- new instance of self.__class__
        """
        new_keys = [key for key in keys_to_keep if key in self]
        new_instance = self.__class__(new_keys, self._features)
        return new_instance

class WordFeatures(namedtuple('WordFeatures', ['key', 'id','tag','freq'])):
    """
    Child of collections.namedtuple. Has fields:
      -- key : str, the word itself
      -- id : int
      -- tag: str, POS tag
      -- freq: how frequent is the word in the training set. Used mostly 
      for comparison (e.g. more frequent vs. less frequent), not in the 
      absolute sense (where it has no meaning, really).
    """
    __slots__ = () # to save memory
    def __str__(self):
        return "word: %s, POS tag: %s"%(self.key, self.tag)

class VocabParams(object):
    """
    Class to keep all the params for creating a vocabulary.
    Class variables:
      -- POS : List[str], list of parts of speech used. 
      Default is ['NNP','NN','NNS','JJ'], for proper nouns, nouns, 
      nouns plural and adjectives.
      -- PROPER_NOUN : str = 'NNP', constant used to indicate proper nouns,
      which stand out for our problem
      -- MIN_WORD_LEN : int, min # of chars in a word to be kept in the vocab
      -- MIN_PROP_NOUN_LEN: int, min # of chars in a proper noun in the vocab
      -- MIN_FREQ : int, min # of times the word should appear in 
      the text corpus the vocab is created from to be kept in the vocab 
      (i.e. words appering less than MIN_FREQ times will be dropped)
      -- Values : namedtuple, has fields 
      'min_word_len', 'min_prop_noun_len','min_freq'. It is used to be able to 
      change the default values when passing the parameters into 
      methosd with construct the vocabulary.
    """
    POS = ['NNP','NN','NNS','JJ']
    PROPER_NOUN = 'NNP'

    MIN_WORD_LEN = 4
    MIN_PROP_NOUN_LEN = 3
    MIN_FREQ = 4

    Values = namedtuple('Params', ['min_word_len', 'min_prop_noun_len','min_freq'] )
    Values.__new__.__defaults__ = (MIN_WORD_LEN,MIN_PROP_NOUN_LEN,MIN_FREQ)

class Vocab(_SortedContainer):
    """
    Child of _SortedContainer.
    Class variables:
      -- feature_cls_name = "WordFeatures", name of the class
      controling word features.
    Instance methods:
      -- proper_nouns(self) : returns a list of all contained proper nouns
    """
    feature_cls_name = "WordFeatures"

    def __init__(self, *args, **kwargs):
        """ Initialize empty vocabulary"""
        super().__init__(*args, **kwargs)

        self.N_proper_nouns = self._count_proper_nouns()

    def _get_by_index(self, index):
        try:
            return np.array(self.sorted_keys)[index]
        except IndexError:
            print("Key not found")
            pass

    def _count_proper_nouns(self):
        count = 0
        for word in self.sorted_keys:
            if self[word].tag == VocabParams.PROPER_NOUN:
                count += 1
        return count

    def proper_nouns(self):
        """
        Output:
          -- List[str], list of all proper nouns in self
        """
        PN = [key for key in self if self[key].tag == VocabParams.PROPER_NOUN]
        return PN

    def _add_new(self, key, feature):
        assert feature.key == key
        super()._add_new(key, feature)
        if feature.tag == VocabParams.PROPER_NOUN:
            self.N_proper_nouns += 1

    def _get_updated_feature(self, key, other_feature):
        assert key in self
        new_freq = self[key].freq + other_feature.freq
        updated_feature = self[key]._replace(freq = new_freq)
        return updated_feature

class PhraseFeatures(namedtuple('PhraseFeatures', ['key', 'id','tag','mutual_info','freq'])):
    """
    Child of collections.namedtuple. Has fields:
      -- key : (str,str), the phrase itself
      -- id : int
      -- tag: (str,str), a pair of strings (POS tags) from VocabParams.POS
      -- mutual_info: float, the value of the 
      mutual information statistic between the two words in the phrase. 
      Computed in the process of creation from training data (text corpus).  
      -- freq: how frequent is the phrase in the training set. Used mostly 
      for comparison (e.g. more frequent vs. less frequent).
    """
    __slots__ = ()
    def __str__(self):
        return "Phrase: %s %s, POS tags: %s, %s"%(*self.key, *self.tag)

class Phrases(_SortedContainer):
    """
    Child of _SortedContainer.
    Class variables:
      -- feature_cls_name = "PhraseFeatures", name of the class
      controling phrase features.
    Instance methods:
      -- to_vocab(self) : returns a new Vocab object made out of self.
    """
    feature_cls_name = "PhraseFeatures"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_vocab(self):
        """
        Returns a new Vocab object, with the keys and features obtained
        from the features of self. Each phrase '(word1,word2)' gets 
        converted to the word 'word1_word2'.
        """
        cls = self.__class__
        to_word_feature = lambda key: cls._as_word_feature(self[key])
        new_keys = [cls._as_word(key) for key in self]
        new_features_list = [to_word_feature(key) for key in self]
        new_features = dict(zip(new_keys,new_features_list))
        vocab = Vocab(new_keys, new_features)
        return vocab

    @staticmethod
    def _as_word(key):
        """
        Input:
          -- key : (str, str), pair of words (word1,word2)
        Output:
          -- the string 'word1_word2'
        """
        assert isinstance(key,tuple)
        SPACE = '_'
        return key[0]+SPACE+key[1]

    @classmethod
    def _as_word_feature(cls, feature):
        """
        Input:
          -- feature : PhraseFeatures
        Output:
          -- WordFeatures, with the field values from PhraseFeatures.
        """
        new_key = cls._as_word(feature.key)
        word_feature = WordFeatures(new_key, feature.id, feature.tag, feature.freq)
        return word_feature

class MSCFeatures(namedtuple('MSCFeatures', ['key', 'id', 'description'])):
    """
    Child of collections.namedtuple. Has fields:
      -- key : str, the MSC code
      -- id : int
      -- description: str, the name of the MSC class.
    """
    __slots__ = ()
    def __str__(self):
        return "Code: %s, description: %s"%(self.key, self.description)

class MSC(_SortedContainer):
    """
    Child of _SortedContainer.
    Class variables:
      -- feature_cls_name = "MSCFeatures", name of the class
      controling MSC code features.
    Class variables:
      -- POSSIBLE_CODE_LENGTHS : List[int], defaults to [2,3,5]
    """
    POSSIBLE_CODE_LENGTHS = [2,3,5]
    feature_cls_name = "MSCFeatures"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, filename):
        """
        Input:
          -- filename: str or int. If int, it should be 
          an element of cls.POSSIBLE_CODE_LENGTHS. Then it loads a 
          default MSC container with codes of the prescribed length.
        """
        # if the filename is just an int indicating the code depth
        if isinstance(filename,int):
            code_length = filename
            assert code_length in MSC.POSSIBLE_CODE_LENGTHS
            filename = cls._get_filename(code_length)

        DTYPE = {0:'str',1:'int',2:'str'}
        return super().load(filename, dtype=DTYPE)

    @staticmethod
    def _get_filename(code_length):
        return MSC_FILENAME_TEMPLATE%code_length

class TFIDFFeatures(namedtuple('TFIDFFeatures', ['key', 'id','tag','idf','tfidf'])):
    """
    Child of collections.namedtuple. Has fields:
      -- key : str, a word
      -- id : int
      -- tag: str or a tuple (str,str), POS tag. If the word is of the form 
      "word1_word2", i.e. obtained from a phrase, the POS is a pair of POS 
      tags, one for each of the words.
      -- idf: the IDF score, computed from a training data. 
      Used only for comparing words with each other (for example 
      at the stage of feature selection).
      -- tfidf: the TFIDF score, computed from a training data. 
      Used for comparing words with each other.
    """
    __slots__ = ()
    def __str__(self):
        scores = "idf: {:.3f}, tfidf: {:.3f}".format(self.idf, self.tfidf)
        info = "word: %s, POS tag: %s, "%(self.key,self.tag)
        return info+scores

class TFIDF(_SortedContainer):
    """
    Child of _SortedContainer.
    Class variables:
      -- feature_cls_name = "TFIDFFeatures", name of the class
      controling the TFIDF features.
    Instance methods:
      -- get_N_lowest_no_NNP(self, N, attr='tfidf') : 
      returns a list of N non-proper-nouns with the lowest value of 
      the attribute. By default, it's 'tfidf'.
      -- get_N_lowest_NNP, get_N_lowest, get_N_highest: similar 
      to get_N_lowest_no_NNP and have the same signature.
      -- proper_nouns(self) : returns a list of all contained proper nouns
    """
    
    feature_cls_name = "TFIDFFeatures"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_N_lowest_no_NNP(self, N, attr='tfidf'):
        """
        Returns a list of N non-proper-nouns with the lowest value of 
        the attribute. By default, it's 'tfidf'.
        Input:
          -- N : int, number of words to select
          -- attr : str, the attribute whose value ot consider
        Output:
          -- List[str], list of corresponding keys
        """
        sorted_by_attr = self._get_sorted_by_attr(attr=attr)
        not_nnp = lambda key: self[key].tag != VocabParams.PROPER_NOUN
        no_NNP = [key for key in sorted_by_attr if not_nnp(key)]
        return no_NNP[-N:]

    def get_N_lowest_NNP(self, N, attr='tfidf'):
        """
        Returns a list of N proper nouns with the lowest value of the attribute.
        Input:
          -- N : int, number of words to select
          -- attr : str, the attribute whose value ot consider
        Output:
          -- List[str], list of corresponding keys
        """
        sorted_by_attr = self._get_sorted_by_attr(attr=attr)
        nnp = lambda key: self[key].tag == VocabParams.PROPER_NOUN
        NNPs = [key for key in sorted_by_attr if nnp(key)]
        return NNPs[-N:]

    def get_N_lowest(self, N, attr='tfidf'):
        """
        Returns a list of N words with the lowest value of the attribute.
        Input:
          -- N : int, number of words to select
          -- attr : str, the attribute whose value ot consider
        Output:
          -- List[str], list of corresponding keys
        """
        sorted_by_attr = self._get_sorted_by_attr(attr=attr)
        return sorted_by_attr[-N:]

    def get_N_highest(self, N, attr='tfidf'):
        """
        Returns a list of N words with the highest value of the attribute.
        Input:
          -- N : int, number of words to select
          -- attr : str, the attribute whose value ot consider
        Output:
          -- List[str], list of corresponding keys
        """
        sorted_by_attr = self._get_sorted_by_attr(attr=attr)
        return sorted_by_attr[:N]

    def _get_sorted_by_attr(self, attr):
        attr_key = lambda w: getattr(self[w],attr)
        sorted_by_attr = sorted( self.sorted_keys, key=attr_key, reverse = True )
        return sorted_by_attr

    def proper_nouns(self):
        """
        Output:
          -- List[str], list of all proper nouns in self
        """
        PN = [key for key in self if self[key].tag == VocabParams.PROPER_NOUN]
        return PN
