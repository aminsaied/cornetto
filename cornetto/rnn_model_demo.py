import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers import l2_regularizer as l2
from tensorflow.contrib.framework import arg_scope

softmax = tf.nn.sparse_softmax_cross_entropy_with_logits # what is the pythonic way to do this?

import datasets
from containers import MSC
from data_handlers import RNNTrainingData
from prediction import Prediction

# set demonstration parameters
n_epochs      = 10
n_inputs      = 70
n_outputs     = 5
n_hidden      = 50
n_neurons     = 100
n_steps       = 20
batch_size    = 64
learning_rate = 0.0003
k_prob        = 0.5
reg_param     = 0.3
selection     = ['05','19','47','57','60']

DIR      = "/tmp/"
filename = "demo_model.ckpt"


def load_rnn_td_builder(w2v_model):
    """Loads object of class RNNTrainingData to construct model inputs."""
    msc_bank = MSC.load(2)
    data = RNNTrainingData(w2v_model, msc_bank, selection, n_steps)
    return data

def batches(input_, length_, output_, batch_size):
    """
    A generator for rnn minibatches.
    """
    m = len(output_)

    for k in range(m // batch_size):

        X_batch      = input_[k * batch_size: (k+1) * batch_size]
        length_batch = length_[k * batch_size: (k+1) * batch_size]
        y_batch      = output_[k * batch_size: (k+1) * batch_size]

        yield X_batch, length_batch, y_batch

def train_model():
    """
    The execution phase: uses the training data just curated to train the model.
    """
    init.run()
    train = rnn_training_data.training
    test  = rnn_training_data.test

    for epoch in range(n_epochs):
        for X_b, length_b, y_b in batches(train.X, train.length, train.Y, batch_size):
            sess.run(training_op, feed_dict={X:X_b, seq_length:length_b, y:y_b})

        # if epoch % 10 == 0:
        training_acc = accuracy.eval(feed_dict={X:train.X, seq_length:train.length, y:train.Y})
        test_acc     = accuracy.eval(feed_dict={X:test.X , seq_length:test.length , y:test.Y })
        _see_evaluation(epoch, training_acc, test_acc)

    save_path = saver.save(sess, DIR+filename)
    return

def _see_evaluation(epoch, training_acc, test_acc):
    """Prints the training and test accuracies at the given epoch."""
    print (("Epoch {:2}: Training acc: {:.2f}, Test acc: "
            "{:.2f}".format(epoch+1, training_acc*100, test_acc*100) ))

def demo_examples(rnn_td_builder):
    """
    Runs through some example abstracts, printing the
    word-by-word predictions of the model.
    """
    examples = datasets.demonstration_examples('rnn')
    for idx, row in examples.iterrows():
        print ("Example", idx+1)
        sentence, codes, description = row
        print (description)
        _presentation(sentence, msc_selection, rnn_td_builder)
        print ("Press enter.")
        input()

def _presentation(sentence, msc_bank, rnn_td_builder):
    """
    Prints the word-by-word predictions of the trained model on a
    given abstract.
    -- sentence: list, of words
    -- msc_bank: MSC, see 'containers'
    -- rnn_td_builder: RNNTrainingData, see 'data_handlers'
    """
    VIEW_LENGTH = 15
    sentence = sentence[:VIEW_LENGTH]
    sentence_rnn = rnn_td_builder._build_rnn_input(sentence)
    X_, length_ = np.array([sentence_rnn]), np.array([VIEW_LENGTH])

    valid_words = [word for word in sentence if word in w2v_model]

    with tf.Session() as sess:
        saver.restore(sess, DIR+filename)
        outputs = rnn_outputs.eval(feed_dict={X:X_, seq_length:length_})
        res = logits.eval(feed_dict={states:outputs[0]})

    pred_indices = np.argmax(res, axis=1)
    code_of = lambda index: msc_bank[int(index)]
    predictions = list(map(code_of, pred_indices))
    preds = predictions[:len(valid_words)]

    sentence_history = pd.DataFrame(preds, valid_words, columns=['prediction'])

    print( sentence_history )

if __name__ == '__main__':

    w2v_model = datasets.load_word2vec(dim=n_inputs)
    rnn_td_builder = load_rnn_td_builder(w2v_model)
    msc_selection = rnn_td_builder.msc_bank

    print ("In this demonstration, for the sake of expediency,")
    print ("we restrict attention to the following five codes:")
    print (msc_selection)

    print("\nWe first curate the data with which to train our network.")
    arxiv = datasets.load_arxiv(depth=2)
    rnn_training_data = rnn_td_builder.build(arxiv)
    print("Done.")

    print ("\nAssembling tensorflow computation graph for the RNN model.")
    print ("(Again for the sake of expediency, we used a stripped down")
    print ("version of our model with only one recurrent layer)")
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs))
    y = tf.placeholder(tf.int64, shape=(None))
    seq_length = tf.placeholder(tf.int64, shape=(None))

    cell_factory = GRUCell(num_units=n_neurons)
    cell_drop    = DropoutWrapper(cell_factory, k_prob)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell_drop, X, dtype=tf.float32,
                    sequence_length=seq_length)

    with arg_scope([fc], weights_regularizer=l2(reg_param)):
        hidden = fc(states, n_hidden)
        logits = fc(hidden, n_outputs, activation_fn=None)

    xentropy = softmax(labels=y, logits=logits)
    base_loss = tf.reduce_mean(xentropy)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost = tf.add_n([base_loss] + reg_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cost)

    correct = tf.nn.in_top_k(logits, y, 2)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    print("Done.")

    print ("\nTraining the model.")
    print ("Effort has been taken to allow time for you to put a brew on.")
    with tf.Session() as sess:
        train_model()

    print ("\nEXAMPLES")
    print(("In the following examples we can see the brains "
           "of the model in action!\n"))
    demo_examples(rnn_td_builder)
