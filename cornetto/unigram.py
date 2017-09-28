import numpy as np
import pandas as pd
from text_processor import WordSelector
from text_to_vec_builders import T2VFromVocab
from data_handlers import UnigramTrainingData
from containers import Vocab, MSC
from prepare_data import MSCCleaner
from prediction import Prediction

EXT = '.pkl'
VOCAB_POSTFIX = '_vocab'
MSC_POSTFIX = '_msc'
DELIMITER = "|"
N_GUESSES = 3

class Unigram(object):

    def __init__(self, vocab, msc_bank, topics=None):
        """
        Topics is a len(msc_bank)-by-len(vocab) matrix with rows word distributions.
        """
        self.vocab = vocab
        self.msc_bank = msc_bank
        self.t2v = T2VFromVocab(vocab)
        self.topics = topics

    def save(self, filename):
        self.vocab.save(filename+VOCAB_POSTFIX)
        self.msc_bank.save(filename+MSC_POSTFIX)
        np.savetxt(filename+EXT, self.topics, delimiter=DELIMITER)

    @classmethod
    def load(cls, filename):
        vocab = Vocab.load(filename+VOCAB_POSTFIX)
        msc_bank = MSC.load(filename+MSC_POSTFIX)
        topics = np.loadtxt(filename+EXT, delimiter=DELIMITER)
        return cls(vocab, msc_bank, topics)

    def train(self, data):
        '''
        -- data: UnigramTrainingData object
        '''
        self.topics = self._build_topics_matrix(data.training)

        accuracy = self._accuracy(data.test.X, data.test.Y)
        print("Training complete.")
        print("Accuracy: {:.2f}%".format(accuracy*100))

    def _build_topics_matrix(self, training_data):
        abstracts, codes = training_data.X.tolist(), training_data.Y.tolist()
        M = np.zeros((len(self.msc_bank), len(self.vocab)))
        for abstract, list_of_codes in zip(abstracts, codes):
            for code in list_of_codes:
                try:
                    msc_ind = self.msc_bank[code].id
                    for word in abstract:
                        vocab_ind = self.vocab[word].id
                        M[msc_ind][vocab_ind] += 1
                except (KeyError, TypeError, IndexError):
                    pass

        M += np.ones(M.shape)

        M_counts = M.sum(axis=1)
        return M/M_counts[:,np.newaxis]

    def predict(self, sentence):
        as_vec = self.t2v.convert_sentence(sentence)
        pred_vec = self._prediction_from_array(as_vec)
        return pred_vec

    def _prediction_from_array(self, X, n_guesses = N_GUESSES):
        model_output = np.matmul(self.topics, X)
        prediction = Prediction(model_output)
        P = prediction.most_likely(n_guesses)
        return P

    def _accuracy(self, X, Y):
        '''
        Returns the accuracy of the network on the given PairedData
        -- data: UnigramTrainingData object
        '''
        P = self._prediction_from_array(X)
        return np.mean(np.sum(np.multiply(P, Y), axis=0) > 0)

if __name__ == "__main__":

    DEPTH = 2
    FILENAME = '-unigram_model_DEPTH' + str(DEPTH)
    TRAINING_DATA_DIR = '-unigram_training_data_DEPTH' + str(DEPTH)

    arxiv = pd.read_pickle('-arxiv_processed.pkl')
    vocab = Vocab.load('-vocab')
    msc_bank = MSC.load(DEPTH)

    X = arxiv['Abstract']
    Y = MSCCleaner._specify(arxiv['MSCs'], depth=DEPTH)

    unigram_data = UnigramTrainingData.build(X, Y, vocab, msc_bank)

    # NB. would like to save this training data, but there is some bug.
    # unigram_data.save(TRAINING_DATA_DIR)
    # print ("Unigram training data saved: " + TRAINING_DATA_DIR + EXT)

    print ("Training model...")
    model = Unigram(vocab, msc_bank)
    model.train(unigram_data)

    print ("Saving model...")
    model.save(filename=FILENAME)
    print ("Model saved: " + FILENAME + EXT)
