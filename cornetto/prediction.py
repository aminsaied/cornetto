import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

import datasets
from containers import MSC
from data_handlers import RNNTrainingData

class ThresholdPrediction(object):
    """
    Model to make predictions based on a threshold-selection method. Concretely,
    model selects a class whenever the probabiliy assigned is greater than a
    certain threshold value. The threshold parameters are selected to optimise
    average F1-score.
    """

    def __init__(self, msc_bank):
        """
        Note that before the model is fit to data, the initial threshold
        vector is full of ones. In this way, before the ThresholdPrediction
        object is fit to data, the 'predict' method simply selects the
        single class with the higghest probability.
        -- msc_bank: MSC, see 'containers'
        """
        self.msc_bank = msc_bank
        self.threshold = np.ones(len(msc_bank))

    def fit(self, p_hat, y, verbose=False):
        """
        Given an array of p_hat and a matrix of labels y, fits
        the threshold parameter vector to maximise the F1 score.
        -- p_hat: 2D array, cols = probability vectors
        -- y: 2D array, cols = vector of labels
        """
        T = np.linspace(0,1)
        K = len(self.msc_bank)
        m = len(T)

        A = np.zeros((K, m))
        for i in range(m):
            y_hat_gaps = (p_hat > T[i]).astype(int)
            y_hat = self._fill_gaps(p_hat, y_hat_gaps)
            A[:,i] = [f1_score(y[i], y_hat[i]) for i in range(K)]

        self.threshold = T[np.argmax(A, axis=1)]

        if verbose:
            f1_scores = A[range(K),np.argmax(A, axis=1)]
            f1_mean = np.mean(f1_scores)
            print("F1 score: {:.2}".format(f1_mean))

    def predict(self, p_hat):
        """
        Uses threshold parameters to predict labels from a given
        probability vector. Namely, class i is assigned whenever
        p_hat[i] > self.threshold[i]
        -- p_hat: 2D array, cols = probability vectors
        """
        y_hat = (p_hat > self.threshold.reshape(-1,1)).astype(int)
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

class Prediction(object):
    """
    Matches a probability vector with the given labels.
    """
    def __init__(self, p_hat, output_dist=[], output_joint_dist=[]):
        """
         -- p_hat: 2D numpy array, cols = probability vectors
         -- output_dist: 1D numpy array, probability distribution of output
         -- output_joint_dist: 2D array, probability of one output given another
        """
        self.p_hat = p_hat
        if len(output_dist)>0:
            self.p_hat = self._reweight_with_dist(self.p_hat, output_dist)
        if len(output_joint_dist)>0:
            self.p_hat = self._reweight_with_joint_dist(self.p_hat, output_joint_dist)

    @staticmethod
    def _reweight_with_dist(p_hat, dist):
        """
        Reweights the probability distribution according to a prior distribution.
        -- p_hat: 2D numpy array, cols = probability vectors
        -- dist: 1D numpy array, probability distribution of output
        """
        return np.multiply(p_hat, dist)

    @staticmethod
    def _reweight_with_joint_dist(p_hat, output_joint_dist):
        """
        Reweights the probability distribution according to a prior distribution.
        -- p_hat: 2D numpy array, cols = probability vectors
        -- output_joint_dist: 2D array, probability of one output given another
        """
        np.fill_diagonal(output_joint_dist, 1)
        max_indices = np.apply_along_axis(np.argmax, 0, p_hat)
        reweight_matrix = output_joint_dist[:,max_indices]
        return np.multiply(reweight_matrix, p_hat)

    def most_likely(self, k=1):
        """Returns matrix of 0 and 1, with 1 at the most likely"""
        indices = self._most_likely_indices(k)
        selected = self._create_mask(indices)
        return selected

    def _most_likely_indices(self, k=1):
        """ Returns UNSORTED array of indices of the k most likely labels. """
        top_k_ind_by_col = partial(Prediction._top_k_ind_in_col, k=k)
        indices_by_col = np.apply_along_axis(top_k_ind_by_col, 0, self.p_hat)
        indices = Prediction._reformat(indices_by_col)
        return indices

    def _create_mask(self, indices):
        shape = self.p_hat.shape
        mask = np.zeros(shape)
        mask[indices] = 1.0
        return mask

    @staticmethod
    def _top_k_ind_in_col(column, k=1):
        indices = np.argpartition(column, -k)[-k:]
        return indices

    @staticmethod
    def _reformat(indices_by_col):
        """
        Makes a list of column-wise indices into a matrix,
        0-th row are x-coordinates, 1-st are y-coordinates
        -- indices_by_col : k x M matrix, each col
                consists of indices in this column when viewed as list.
        """
        size = len(indices_by_col)
        xcoord, ycoord = [], []
        for (x,y), value in np.ndenumerate(indices_by_col):
            xcoord.append( value )
            ycoord.append( y )
        stacked = np.stack((xcoord, ycoord))
        # convert to list for numpy indexing specifics reason
        return list(stacked)
