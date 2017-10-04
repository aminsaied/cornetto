import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, fbeta_score

import datasets
from containers import MSC
from data_handlers import RNNTrainingData

DEFAULT_THRESHOLD = 0.2

class ThresholdPrediction(object):
    """
    Model to make predictions based on a threshold-selection method. Concretely,
    model selects a class whenever the probabiliy assigned is greater than a
    certain threshold value. The threshold parameters are selected to optimise
    average F1-score.
    """

    def __init__(self, dim=0, threshold=None, labels=None):
        """
        Note that before the model is fit to data, the initial threshold
        vector is full of ones. In this way, before the ThresholdPrediction
        object is fit to data, the 'predict' method simply selects the
        single class with the higghest probability.
        -- K: either the number of classes (=dim of the vector), 
        or a 'collection' of classes, i.e. anything we can take 'len' of. 
        """
        assert isinstance(dim, int)
        self.dim = dim
        self.threshold = threshold
        self.labels = labels

    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value):
        if value is None:
            value = np.full(self.dim, DEFAULT_THRESHOLD)
        value_as_arr = np.array(value).reshape(-1,1)
        assert value_as_arr.shape[0] == self.dim
        self._threshold = value_as_arr
        
    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self, value):
        if value is None:
            value = np.array(range(self.dim), dtype=str)
        value_as_arr = np.array(value).reshape(-1,1)
        assert value_as_arr.shape[0] == self.dim
        self._labels = value_as_arr
        
    def fit(self, p_hat, y, refinement=50, beta=1.0, verbose=False):
        """
        Given a 2D array p_hat, each column a probability vector, 
        and a matrix of labels y, fits the threshold parameter 
        vector to maximise the F-beta score.
        Input:
          -- p_hat: 2D array, cols = probability vectors
          -- y: 2D array, cols = vectors of labels (vectors of 0's and 1's)
          -- refinement : int, the number of threshold values to consider
          -- beta : float, weight of precision in harmonic mean
          -- verbose : bool, if True prints the average f-beta score.
        """
        test_thresholds = np.linspace(0,1, num=refinement)

        A = np.zeros((self.dim, refinement))
        for i, threshold in enumerate(test_thresholds):
            threshold_vec = np.full(self.dim, threshold)
            y_hat = self._predict_with_threshold(p_hat, threshold_vec)
            A[:,i] = fbeta_score(y.T,y_hat.T,beta=beta,average=None)

        self.threshold = test_thresholds[np.argmax(A, axis=1)]

        if verbose:
            f_scores = A[range(self.dim),np.argmax(A, axis=1)]
            f_mean = np.mean(f_scores)
            message = "Average F-beta score with beta={:.2} is: {:.2}"
            print(message.format(beta, f_mean))

    def predict(self, p_hat):
        """
        Uses threshold parameters to predict labels from a given
        probability vector. Namely, class i is assigned whenever
        p_hat[i] > self.threshold[i]
        -- p_hat: 2D array, cols = probability vectors
        """
        y_hat = self._predict_with_threshold(p_hat, self.threshold)
        return y_hat
    
    def save(self, filename):
        """
        Input:
          -- filename: str, filename with a path, *no extension*, 
          i.e. 'my_file' isnstead of 'my_file.txt'
        """
        EXTENSION = '.csv'
        data = np.hstack( (self.threshold, self.labels) )
        pred_as_df = pd.DataFrame(data)
        pred_as_df.to_csv(filename+EXTENSION, mode = 'w', index=False)
        
    @classmethod
    def load(cls, file, encoding = 'ISO-8859-1'):
        """
        Input:
           -- file : either a filename or a file object.
           -- dtype : dictionary of types for each feature attribute
        """
        DTYPE = {0:'float', 1:'str'}
        # if file is just a filename
        if isinstance(file, str):
            EXTENSION = '.csv'
            file += EXTENSION
        try:
            df = pd.read_csv(file, dtype=DTYPE, encoding=encoding)
        except (ValueError, EOFError):
            print("Can't read the file.")
            empty = cls()
            return empty
        dim = len(df.index)
        return cls(dim, df['0'], df['1'])
    
    def _predict_with_threshold(self, p_hat, T):
        p_hat = p_hat.reshape(self.dim, -1)
        T = T.reshape(self.dim, -1)
        y_hat = (p_hat > T).astype(int)
        y_hat_no_gaps = self._fill_gaps(p_hat, y_hat)
        return y_hat_no_gaps
        
    @staticmethod
    def _fill_gaps(p_hat, y_hat):
        """
        If y_hat has a column of zeros, i.e., if there is an instance
        with no prediction, predicts the highest probability occuring,
        (even though less than threshold).
        """
        n_pred = np.sum(y_hat, axis=0)
        pred_max = np.argmax(p_hat, axis=0)

        gap_indices = np.where( n_pred == 0 )[0]
        code_indices = [pred_max[i] for i in gap_indices]

        y_hat[code_indices, gap_indices] += 1
        return y_hat


class ThresholdPredictionSimple(object):
    """
    A simple prediction model, that selects a single threshold value t
    maximising F1-score.
    """
    def __init__(self, p_hat):
        """
        -- p_hat: array, vector of p_hat
        """

        self.p_hat = p_hat
        self.threshold = 0.5

    def y_hat():
        y_hat = (self.p_hat > self.threshold).astype(int)
        return y_hat

    def max_threshold(self, y, verbose=False):
        """
        -- y: array, m-by-K matrix of one_hot codes
                     rows = instances (m instances)
                     cols = msc_codes (K classes)
        """
        thresholds = np.linspace(0, 1, 30)
        f1s = [self._compute_f1(y, t) for t in thresholds]
        index_max = np.argmax(f1s)
        t = thresholds[index_max]
        f1_max = f1s[index_max]

        if verbose:
            plt.plot(thresholds, f1s, '-')
            plt.xlabel('Thresholds')
            plt.ylabel('F1-score')
            plt.show()
            print ("Max F1-score: {:.2}, with threshold of {:.2}".format(f1_max,t))

        self.threshold = t

    def _compute_f1(self, y, threshold):

        y_hat = self._compute_y_hat(threshold)

        K = y.shape[1]
        f1s = [f1_score(y[:,i], y_hat[:,i]) for i in range(K)]
        f1 = np.mean(f1s)
        return f1

    def _compute_y_hat(self, threshold):
        """Uses the threshold to make predictions based on RNN output."""
        y_hat = (self.p_hat > threshold).astype(int)
        y_hat_no_gaps = self._fill_gaps(self.p_hat, y_hat)
        return y_hat_no_gaps

    @staticmethod
    def _fill_gaps(p_hat, y_hat):
        """
        If y_hat has a row of zeros, i.e., if there is an instance
        with no prediction, predicts the highest probability occuring,
        (even though less than threshold).
        """
        n_pred = np.sum(y_hat, axis=1)
        pred_max = np.argmax(p_hat, axis=1)

        gap_indices = np.where( n_pred == 0 )[0]
        code_indices = [pred_max[i] for i in gap_indices]

        y_hat[gap_indices, code_indices] += 1
        return y_hat
