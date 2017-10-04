import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers import l2_regularizer as l2
from tensorflow.contrib.framework import arg_scope

softmax = tf.nn.sparse_softmax_cross_entropy_with_logits # what is the pythonic way to do this?

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import modules.datasets
from modules.data_handlers import RNNTrainingData

def default_hyper_params():
    """Assigns default hyperparameters."""
    N_NEURONS = 100
    N_HIDDEN = 100
    N_STEPS = 50
    K_PROB = 0.9
    return N_NEURONS, N_HIDDEN, N_STEPS, K_PROB

def default_training_params():
    """Assigns default training parameters."""
    N_EPOCHS = 100
    BATCH_SIZE = 64
    EPSILON = 0.0001
    return N_EPOCHS, BATCH_SIZE, EPSILON

def load_data():
    """
    Loads preprocessed arxiv data and RNN training data.
    """
    print ("Loading the arxiv.")
    arxiv = datasets.load_arxiv(depth=2)

    print ("Loading the RNN training data.")
    print ("Select depth (2/3/5):")
    depth = int(input())
    rnn_training_data = datasets.training_data("rnn", depth=depth)

    n_inputs = rnn_training_data.training.X.shape[2] # dimension of w2v model
    n_outputs = rnn_training_data.training.dimY

    return arxiv, rnn_training_data, n_inputs, n_outputs

def set_use_defaults():
    """
    If True will use our default parameters.
    If False user can input parameters.
    """
    print ("Use default parameters to build and train the model? (1/0)")
    use_defaults = bool(input())
    return use_defaults

def set_hyper_params(use_defaults):
    """
    User selects hyperparameters for the model.
    """
    if use_defaults:
        n_neurons, n_hidden, n_steps, k_prob = default_hyper_params()
        return n_neurons, n_hidden, n_steps, k_prob

    print ("Select number of neurons in recurrent layer (default " +
            "100):")
    n_neurons = int(input())
    print ("Select number of hidden neurons in fully connected " +
            "layer (default 100):")
    n_hidden = int(input())
    print ("Select n_steps; the max number of words to be read " +
            "from each abstract (default 50):")
    n_steps = int(input())
    print ("Select k_prob; the dropout probability (default 0.5):")
    k_prob = float(input())

    return n_neurons, n_hidden, n_steps, k_prob

def set_training_params(use_defaults):
    """
    User selects training parameters.
    """
    if use_defaults:
        n_epochs, batch_size, epsilon = default_training_params()
        return n_epochs, batch_size, epsilon

    print ("Select number of epochs to train (default 100):")
    n_epochs = int(input())
    print ("Select batch size (default 64):")
    batch_size = int(input())
    print ("Select learning rate (default 0.0001):")
    epsilon = float(input())
    return n_epochs, batch_size, epsilon

def rnn_minibatches(input_, length_, output_, batch_size):
    """
    A generator for rnn_minibatches.
    """
    m = len(output_)

    for k in range(m // batch_size):

        X_batch      = input_[k * batch_size: (k+1) * batch_size]
        length_batch = length_[k * batch_size: (k+1) * batch_size]
        y_batch      = output_[k * batch_size: (k+1) * batch_size]

        yield X_batch, length_batch, y_batch

def see_evaluation(epoch, training_acc, test_acc):
    """Prints the training and test accuracies at the given epoch."""
    print ("Epoch ", epoch, "Training acc: ", training_acc*100, "Test acc: ", test_acc*100)

if __name__ == '__main__':

    arxiv, rnn_training_data, n_inputs, n_outputs = load_data()

    use_defaults = set_use_defaults()

    n_neurons, n_hidden, n_steps, k_prob = set_hyper_params(use_defaults)

    print ("Building computation graph.")

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    seq_length = tf.placeholder(tf.int64, shape=(None), name='seq_length')
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    reg_param = tf.placeholder_with_default(0.2, tf.float32, shape=())

    with tf.name_scope('recurrent_neurons'):
        cell_factory = GRUCell(num_units=n_neurons)
        cell_drop    = DropoutWrapper(cell_factory, k_prob)
        __, states = tf.nn.dynamic_rnn(cell_drop, X, dtype=tf.float32,
                                        sequence_length=seq_length)

    with tf.name_scope('output'):
        with arg_scope([fc], weights_regularizer=l2(reg_param)):
            hidden = fc(states, n_hidden, scope='hidden')
            logits = fc(hidden, n_outputs, activation_fn=None, scope='logits')

    with tf.name_scope('cost'):
        xentropy = softmax(labels=y, logits=logits)
        base_loss = tf.reduce_mean(xentropy, name='base_loss')
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost = tf.add_n([base_loss] + reg_loss, name='cost')

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(cost)

    with tf.name_scope('accuracy'):
        correct = tf.nn.in_top_k(logits, y, 3)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    print ("Computation graph complete.")

    n_epochs, batch_size, epsilon = set_training_params(use_defaults)

    print ("Training the model.")

    with tf.Session() as sess:

        init.run()

        train = rnn_training_data.training
        test  = rnn_training_data.test

        for epoch in range(n_epochs):

            for X_b, length_b, y_b in rnn_minibatches(train.X, train.length, train.Y, batch_size):

                sess.run(training_op, feed_dict={X:X_b, seq_length:length_b, y:y_b,
                                                learning_rate:epsilon})

            if epoch % 10 == 0:
                training_acc = accuracy.eval(feed_dict={X:train.X, seq_length:train.length, y:train.Y})
                test_acc = accuracy.eval(feed_dict={X:test.X, seq_length:test.length, y:test.Y})
                see_evaluation(epoch, training_acc, test_acc)

        DIR = "/tmp/"
        FILENAME = "my_model_final.ckpt"
        print ("Saving model to", DIR+FILENAME)

        save_path = saver.save(sess, DIR+FILENAME)
