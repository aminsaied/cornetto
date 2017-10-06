# import standard libraries
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

#import methods from libraries
from collections import Counter, namedtuple
from text_processor import WordSelector, TextCleaner, POSTagger
from containers import WordFeatures, Vocab, Phrases, PhraseFeatures, VocabParams, TFIDF, TFIDFFeatures

class VocabBuilder(object):

    def __init__(self, params=VocabParams.Values()):
        self.params = params

    def from_series(self,series):
        """
         -- series : pandas series, collection of text to build vocab only
        """
        corpus = series.tolist()
        return self.from_text_corpus(corpus)

    def from_text_corpus(self, text):
        """
            Given a text as a list of strings,
            builds and returns a list of sorted keys and
            a list of word features used to construct Vocab
        """
        sentences = list(map(TextCleaner.get_clean_words, text))
        return self.from_sentences(sentences)

    def from_sentences(self, sentences):
        """
        Sentences is a list of sentences, each sentence is a
        list of words.
        """
        tagged_words = []

        #TODO: this is really stupid
        for sentence in sentences:
            tagged_words += POSTagger.tag_words(sentence)

        counter = Counter(tagged_words)
        tagged_not_rare = self._get_not_rare(counter)

        word_tags = dict(self._select_from_tagged(tagged_not_rare))

        sorted_words = self._get_sorted_keys(counter, word_tags)
        word_features = VocabBuilder._build_features(counter, word_tags, sorted_words)
        features = dict( zip(sorted_words, word_features) )

        return Vocab(sorted_words, features)

    @classmethod
    def _build_features(cls, counter, word_tags, sorted_words):
        size = len(sorted_words)
        tag = lambda id: word_tags[sorted_words[id]]
        word = lambda id: sorted_words[id]
        freq = lambda id: counter[(word(id),tag(id))]
        feature = lambda id: WordFeatures(word(id),id,tag(id),freq(id))
        word_features = [feature(id) for id in range(size)]
        return word_features

    def _get_not_rare(self, counter):
        not_rare = [elem for elem in counter if counter[elem] >= self.params.min_freq]
        return not_rare

    def _get_sorted_keys(self, counter, word_tags):
        # NOTE: originally it was sorting the words by POS and by frequency
        return list(word_tags)

    def _select_from_tagged(self, tagged_words):
        """
            Given a list of tagged words, select the words of the right POS and length
        """
        selected = self._select_suitable(tagged_words)
        return selected

    def _select_suitable(self, tagged_words):
        """ Selects tagged words of correct POS and length.
            -- tagged_words is a list of pairs (word, tag) """
        return [word_tag for word_tag in tagged_words if self._is_suitable(word_tag) ]

    def _is_suitable(self, word_tag):
        word, tag = word_tag[0], word_tag[1]

        is_correct_pos = (tag in VocabParams.POS)
        is_long = (len(word) >= self.params.min_word_len)
        is_proper_noun = (tag == VocabParams.PROPER_NOUN)
        is_pn_long = (len(word) >= self.params.min_prop_noun_len)
        is_long_proper_noun = is_pn_long and is_proper_noun
        is_long_correct_pos = is_long and is_correct_pos

        if is_long_correct_pos or is_long_proper_noun:
            return True
        else:
            return False

class PhraseBuilder(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def from_text_corpus(self, corpus, percentage=0.7):
        """Returns a list of phrases"""
        word_selector = WordSelector(self.vocab)
        words = word_selector.select_words(corpus)
        return self._from_words(words, percentage)

    def from_sentences(self, sentences, percentage=0.7):
        # flatten and append all sentences to get a list of words
        all_words = [word for sentence in sentences for word in sentence]
        word_selector = WordSelector(self.vocab)
        selected_words = word_selector.select_from_sentence(all_words)
        return self._from_words(selected_words,percentage)

    def _from_words(self, words, percentage=0.7):
        context_count_matrix = self._build_context_counts(words)
        I = self._mutual_information(context_count_matrix)

        N_phrases = int(percentage*len(self.vocab))
        phrases = self._pick_phrases(I, N_phrases)

        sorted_keys = self._get_sorted_keys(phrases)
        features = self._build_features(sorted_keys, I, context_count_matrix)

        return Phrases(sorted_keys, features)

    def _build_context_counts(self, words):
        """Returns context_count_matrix using
        asymmetric context windows of size 2.
            -- words: list of strings
        """
        vocab = self.vocab
        shape = (len(vocab),len(vocab))
        context_count = np.zeros(shape=shape)
        N_contexts = 0
        for prev, nxt in zip( words, words[1:] ):
            if (prev in vocab) and (nxt in vocab):
                i, j = vocab[prev].id, vocab[nxt].id
                context_count[i,j] += 1
                N_contexts += 1
        return context_count

    def _mutual_information(self, context_count_matrix):
        assert len(self.vocab) == context_count_matrix.shape[1]
        counts_vector = np.sum(context_count_matrix, axis = 1)
        N_contexts = np.sum(counts_vector)
        prob_vector = (counts_vector/N_contexts).reshape(len(self.vocab),1)

        JP_array = self._joint_prob(prob_vector, context_count_matrix/N_contexts)
        DP_array = self._disjoint_prob(prob_vector)
        with np.errstate(divide='ignore', invalid='ignore'):
            log = np.log2(np.divide(JP_array, DP_array))
            I =  np.sum( JP_array * log, axis=0)
        return np.nan_to_num(I)

    def _pick_phrases(self, I, N_phrases):
        """Returns dictionary with keys tuples (first_word, second_word)
           built from phrases"""
        first, second = np.unravel_index(np.argsort(I, axis=None)[-N_phrases:], I.shape)
        first_words = self.vocab[list(map(int, first))]
        second_words = self.vocab[list(map(int, second))]

        return list(zip(first_words, second_words))

    def _get_sorted_keys(self, phrases):
        # a separate method in case we want to change the sorting order
        return sorted(phrases)

    def _build_features(self, phrases, mutual_info_matrix, count_matrix):
        tag = lambda word: self.vocab[word].tag
        tags = lambda phrase: (tag(phrase[0]), tag(phrase[1]))

        index = lambda word: self.vocab[word].id
        indices = lambda phrase: (index(phrase[0]), index(phrase[1]))

        mut_info = lambda phrase: mutual_info_matrix[ indices(phrase) ]
        freq = lambda phrase: int(count_matrix[ indices(phrase) ])
        feature = lambda id, phrase: PhraseFeatures(phrase, id, tags(phrase), mut_info(phrase), freq(phrase) )

        features_list = [ feature(id, phrase) for id, phrase in enumerate(phrases) ]
        features = dict( zip(phrases,features_list) )

        return features

    @staticmethod
    def _joint_prob(prob_vector, normalised_context_count_matrix):
        on_on = normalised_context_count_matrix
        on_off = prob_vector - on_on
        off_on = prob_vector.transpose() - on_on
        off_off = 1.0 - prob_vector - off_on
        return np.array( [off_off, off_on, on_off, on_on] )

    @staticmethod
    def _disjoint_prob(prob_vector):
        on, off = prob_vector, 1 - prob_vector
        on_on = np.outer(on, on)
        on_off = np.outer(on, off)
        off_on = np.outer(off, on)
        off_off = np.outer(off, off)
        return np.array( [off_off, off_on, on_off, on_on] )










class TFIDFBuilder(object):
    """
    Given a vocabulary and a list of documents (lists of words), compute the
    text frequence/inverse document frequency (TFIDF) statistic.
    """
    KAPPA = 0.5
    def __init__(self):
        # NOTE: at the moment, there is no need for there to be an __init__
        # method. However, if we want to change the scoring functions,
        # we will have to have it.
        pass

    def build(self, vocab, docs, labels):
        features = self._build_features(vocab, docs, labels)
        return TFIDF(vocab.sorted_keys, features)

    def _build_features(self, vocab, docs, labels):
        text_freq, inv_doc_freq = TFIDFBuilder._compute_tf_and_idf(vocab,docs,labels)
        tfidf_scores = TFIDFBuilder._compute_tfidf(text_freq,inv_doc_freq)

        tag = lambda word: vocab[word].tag
        feature = lambda i: (vocab[i],i,tag(vocab[i]),inv_doc_freq[i], tfidf_scores[i])
        features_tups = [feature(i) for i in range(len(vocab))]
        features_list = [TFIDFFeatures(*tup) for tup in features_tups]
        features_dict = dict(zip(vocab.sorted_keys, features_list))
        return features_dict

    @classmethod
    def _compute_tf_and_idf(cls, vocab, docs, labels):
        text_freq = cls._compute_tf(vocab, docs, labels)
        doc_freq = cls._compute_doc_freq(text_freq)
        # NOTE: here we have chosen a particular idf.
        # TODO: make it so we can change idf method.
        inv_doc_freq = cls._idf_smooth(doc_freq, len(set(labels)))
        return text_freq, inv_doc_freq

    @classmethod
    def _compute_tf(cls, vocab, docs, labels):
        occur_matrix = cls._occurances_matrix(vocab, docs, labels)
        text_freq = cls._tf_matrix(occur_matrix)
        return text_freq

    @staticmethod
    def _occurances_matrix(vocab, docs, labels):
        N_vocab = len(vocab)
        # this is silly, but it makes so that we can record all the labels
        N_labels = max(labels)+1
        occur_matrix = np.zeros((N_vocab, N_labels))

        for index, doc in enumerate(docs):
            j = labels[index]

            for word in doc:
                if word in vocab:
                    i = vocab[word].id
                    occur_matrix[i,j] += 1

        return occur_matrix

    @classmethod
    def _tf_matrix(cls, occur_matrix):
        tf = lambda vec: cls.KAPPA + (1-cls.KAPPA)*(vec / np.max(vec))
        AXIS = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            tf_matrix = np.apply_along_axis(tf,AXIS,occur_matrix)
            where_nan = np.isnan(tf_matrix)
            tf_matrix[where_nan] = 0
        return tf_matrix

    @classmethod
    def _compute_doc_freq(cls, text_freq):
        f = lambda x: x>cls.KAPPA
        doc_freq = np.apply_along_axis(f, 1, text_freq).sum(axis=1)
        return doc_freq

    @classmethod
    def _idf_smooth(cls, doc_freq, N_docs):
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_freq = N_docs/doc_freq
        inv_freq[inv_freq == np.inf] = 0
        inv_doc_freq = np.log(1+inv_freq)
        return inv_doc_freq

    @classmethod
    def _compute_tfidf(cls, text_freq, inv_doc_freq):
        word_freq = np.sum(text_freq, axis=1)
        word_freq_norm = word_freq - np.min(word_freq)
        scores = np.multiply(word_freq_norm, inv_doc_freq)
        return scores
