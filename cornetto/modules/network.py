# uses Python 3.6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data_handlers import T2VTrainingData, PairedData
from prediction import Prediction
from abc import ABC, abstractmethod

class _ActivationFunction(ABC):

    @staticmethod
    @abstractmethod
    def value_at():
        pass

    @abstractmethod
    def derivative_at():
        pass

class Sigmoid(_ActivationFunction):

    @staticmethod
    def value_at(z):
        return 1/(1 + np.exp(-z))

    @classmethod
    def derivative_at(cls,z):
        return cls.value_at(z)*(1-cls.value_at(z))

class Identity(_ActivationFunction):
    @staticmethod
    def value_at(z):
        return z

    @staticmethod
    def derivative_at(z):
        return z

class Softmax(_ActivationFunction):
    @staticmethod
    def value_at(z):
        exp_z = np.exp( z-np.mean(z) )
        return exp_z/np.sum(exp_z)

    @classmethod
    def derivative_at(cls, z):
        return cls.value_at(z) - (cls.value_at(z))**2

class _CostFunction(ABC):

    @staticmethod
    @abstractmethod
    def value_at():
        pass

    @staticmethod
    @abstractmethod
    def grad_at():
        pass

class QuadraticMean(_CostFunction):

    name = "Quadratic Mean"

    @staticmethod
    def value_at(guess, truth):
        diff = guess - truth
        return (1/2)*np.mean(np.sum(diff**2,axis=1))

    @staticmethod
    def grad_at(guess, truth):
        return guess - truth

class NeuronLayer(object):
    """
    Models a simple layer of neurons, together with the activation function
    """
    def __init__(self, dim_in, dim_out, activation_function):
        """
        Models a single layer of neurons.
        -- dim_in, dim_out : int, dimensions of input and output
        -- activation : method, accepts a numpy array and returns a numpy array
        """
        self.dim_in, self.dim_out = dim_in, dim_out
        self.init_params(dim_in, dim_out)
        self.activation_function = activation_function

    def init_params(self, dim_in, dim_out):
        self._weights = np.random.randn(dim_out, dim_in)*(1/np.sqrt(dim_in))
        self._biases = np.zeros((dim_out,1))

    def set_params(self, weights, biases):
        self._weights = weights
        self._biases = biases

    def get_params(self):
        return self._weights, self._biases

    def output_no_activation(self, x):
        """ -- x : numpy array, 1D or 2D input"""
        assert x.shape[0] == self.dim_in # this should probably be try-except combination
        return np.matmul(self._weights, x) + self._biases

    def pullback_with_weight(self, M):
        """Multiplies matrix by weights transposed"""
        return np.matmul(self._weights.transpose(),M)

    def output_with_activation(self, x):
        return self.activation_function.value_at( self.output_no_activation(x) )

    @staticmethod
    def are_compatible_params(weights, biases):
        compatible_weights = (weights.shape == (self.dim_in, self.dim_out))
        compatible_biases = (biases.shape[0] == self.dim_out)
        return compatible_weights and compatible_biases

class Network(object):
    """
    Collection of layers connected with each other
    """
    def __init__(self, layers):#, training_params=None):
        assert Network.are_compatible_layers(layers)
        self.layers = layers
        self.length = len(self.layers)
        self.last_layer = layers[-1]

    @classmethod
    def load(cls, filename):
        layers = pd.read_pickle(filename)
        return cls(layers)

    @staticmethod
    def are_compatible_layers(layers):
        consecutive_pairs = zip(layers, layers[1:])
        for prev, next in consecutive_pairs:
            if prev.dim_out != next.dim_in:
                return False
        return bool(layers) # takes care of layers=[]

    def output_with_activation(self, x):
        prev_output = x
        for layer in self.layers:
            next_output = layer.output_with_activation(prev_output)
            prev_output = next_output
        return next_output

    def save(self, filename):
        '''
        Saves the network parameters into a (.pkl) file.
        return = None
        filename = str, name (!without the extension!) of the target file
        '''
        data = [layer.get_params() for layer in self.layers]
        pd.to_pickle(data, filename + '.pkl')

    def train(self, data, params, see_cost=False):
        '''
        Trains the network on the given training data using backpropogation
        -- data: NetworkTrainingData object
        -- cost_function: subclass of CostFunction
        -- params: TrainingParams object
        '''
        print("Train using the following training parameters.")
        print(params)
        m = params.mini_batch_size
        training_data = data.training

        cost_history = []
        for i in tqdm.tnrange(params.n_iterations):
            for mini_batch in self._mini_batches(training_data, params):
                self._train_mini_batch(mini_batch, params)
            cost = self.compute_cost(data.test.X, data.test.Y,
                                     params.cost_function, params.reg_parameter)
            cost_history.append(cost)
            if (i+1)%5 == 0:
                print("Completed iteration %s"%(i+1))
                training_accuracy = self._accuracy(data.training)
                test_accuracy = self._accuracy(data.test)
                print("-- Training accuracy: {:.4f}%".format(training_accuracy*100))
                print("-- Test accuracy    : {:.4f}%".format(test_accuracy*100))

        print("\nTraining complete.")

        accuracy = self._accuracy(data.test)
        print("Accuracy: {:.4f}%".format(accuracy*100))

        if see_cost:
            return cost_history

    def compute_cost(self, X, Y, cost_function, reg_parameter):
        assert X.ndim == 2
        cost = self._unregularisaed_cost(cost_function, X, Y)
        batch_size = X.shape[1]
        reg = self._regularisation(reg_parameter, batch_size)
        return cost + reg

    def _unregularisaed_cost(self, cost_function, X, Y):
        guess = self.output_with_activation(X)
        return cost_function.value_at(guess, Y)

    def _regularisation(self, lambda_, batch_size):
        reg = 0
        for layer in self.layers:
            for weights in layer._weights:
                    reg += np.multiply(weights, weights).sum()
        return reg * lambda_/(2*batch_size)

    def _accuracy(self, data):
        '''
        Returns the accuracy of the network on the given PairedData
        -- data: PairedData object
        '''
        N_GUESSES = 3 # this is the number of guesses we make
        output = self.output_with_activation(data.X)
        prediction = Prediction(output)
        P = prediction.most_likely(N_GUESSES)
        # P = np.zeros_like(output)
        # P[np.argmax(output, axis=0),range(output.shape[1])] = 1
        Y = data.Y
        return np.mean(np.sum(np.multiply(P, Y), axis=0))

    def plot_cost_history(cost_history):
        plt.plot(cost_history)
        plt.show()

    def _mini_batches(self, training_data, params):
        """Given an object of the class PairedData returns a
        generator of mini_batches"""
        m = params.mini_batch_size
        n_cols = training_data.X.shape[1]
        for i in range(n_cols//m):
            batch_X = training_data.X[:, i*m:(i+1)*m]
            batch_Y = training_data.Y[:, i*m:(i+1)*m]
            yield PairedData(batch_X, batch_Y)

    def _train_mini_batch(self, training_data, params):
        X, Y = training_data.X, training_data.Y
        energies = self._feed_forward(X)
        updates = self._back_propogation(energies, Y, params)
        self._update_parameters(updates, params)

    def _feed_forward(self,X):
        """Given input matrix X returns object of class Energies"""
        energies = Energies()
        energies.add_to_A_stack(X)
        for layer in self.layers:
            Z = layer.output_no_activation(energies.peek_A())
            energies.add_to_Z_stack(Z)
            A = layer.activation_function.value_at(Z)
            energies.add_to_A_stack(A)
        return energies

    def _back_propogation(self, energies, Y, params):
        """Returns object of the class Updates"""
        updates = Updates()
        cost_function = params.cost_function
        cost_grad = cost_function.grad_at(energies.next_A(),Y)
        activation_function = self.last_layer.activation_function
        updates.compute_next(activation_function, energies, cost_grad)

        for layer in reversed(self.layers[1:]):
            delta = updates.last_delta()
            activation_function = layer.activation_function
            updates.compute_next(activation_function, energies, layer.pullback_with_weight(delta))

        return updates

    def _update_parameters(self, updates, params):
        learning_rate = params.learning_rate
        lambda_ = params.reg_parameter
        m = params.mini_batch_size
        reg_factor = 1 - (learning_rate*lambda_/m)
        for index, layer in enumerate(self.layers):
            layer._weights = reg_factor * layer._weights - learning_rate*updates.weights_updates[-index-1]
            layer._biases = layer._biases - learning_rate*updates.biases_updates[-index-1].reshape((layer.dim_out,1))

    ## TODO rename this!
    @staticmethod
    def weird_matrix_operation(M,N):
        m = M.shape[1]
        assert N.shape[1] == m
        M_cols = M.transpose().reshape(M.shape[1],M.shape[0],1)
        N_cols = N.transpose().reshape(N.shape[1],1,N.shape[0])
        temp = np.multiply(M_cols , N_cols)
        return np.mean(temp,axis=0)

    def __str__(self):
        print_string = "Network has %s layers:\n"%self.length
        for index, layer in enumerate(self.layers):
            print_string += "Layer %s: dim_in = %s, dim_out = %s\n"%(index+1, layer.dim_in, layer.dim_out)
        # print_string += "\nTraining parameters:\n"
        # print_string += self.training_params.__str__()
        return print_string

class TrainingParams(object):
    """
    Specify the training parameters of the model.
    -- learning_rate: float, defaults to 0.1
    -- mini_batch_size: int, defaults to 1
    -- cost_function: CostFunction, defaults to QuadraticMean
    """
    def __init__(self, learning_rate=0.1,
                       reg_parameter=0.0001,
                       mini_batch_size=1,
                       n_iterations=50,
                       cost_function=QuadraticMean):
        self.learning_rate = learning_rate
        self.reg_parameter = reg_parameter
        self.cost_function = cost_function
        self.mini_batch_size = mini_batch_size
        self.n_iterations=n_iterations

    def __str__(self):
        return "Learning rate  : %s\nCost Function  : %s\nMini-batch size: %s\nReg param      : %s\n"\
                %(self.learning_rate, self.cost_function.name, self.mini_batch_size, self.reg_parameter)

class Energies(object):
    """
    Data structure holding two stacks of activation energies, both
    with and without activation function
    """
    def __init__(self):
        self.Z_stack = []
        self.A_stack = []
    def add_to_Z_stack(self, Z):
        self.Z_stack.append(Z)
    def add_to_A_stack(self, A):
        self.A_stack.append(A)
    def next_Z(self):
        return self.Z_stack.pop()
    def next_A(self):
        return self.A_stack.pop()
    def peek_A(self):
        return self.A_stack[-1]

class Updates(object):
    """
    Holds updates to weights and biases for the whole network
    """
    def __init__(self):
        self.biases_updates = []
        self.weights_updates = []
        self.deltas = []

    def add_delta(self, delta):
        self.deltas.append(delta)

    def last_delta(self):
        return self.deltas[-1]

    def compute_next(self, activation_function, energies, temp):
        f_prime = activation_function.derivative_at(energies.next_Z())
        delta = np.multiply(temp ,f_prime)
        delta_b_mean = np.mean(delta, axis=1)
        delta_W_mean = Network.weird_matrix_operation(delta, energies.next_A())
        self.deltas.append(delta)
        self.biases_updates.append(delta_b_mean)
        self.weights_updates.append(delta_W_mean)
