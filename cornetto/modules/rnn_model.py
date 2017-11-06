
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers import l2_regularizer as l2
from tensorflow.contrib.framework import arg_scope
import json
import functools

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data_handlers import RNNTrainingData

def property_with_check(input_fn):
    attribute = '_cache_' + input_fn.__name__
    
    @property
    @functools.wraps(input_fn)
    def check_attr(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, input_fn(self))
        return getattr(self, attribute)
    
    return check_attr

class RNNModel(object):
    
    TF_EXT = '.ckpt'
    DATAEXT = '.csv'
    PATH = './data/models/rnn/'
    
    def __init__(self,
                 n_inputs, # dim of w2v model,
                 n_outputs, # total number of labels,
                 n_steps = 50, # max num of words in a sequence (abstract)
                 n_units = 3, # num of units in LSTM unit
                 n_hidden_neurons = 100, # neurons in the fully connected layer
                 k_prob = 0.6, # dropout probability
                 save_file = None,
                 w2v_model = None
                 ):
        
        tf.reset_default_graph()
        
        self.params = {'n_inputs' : n_inputs,
                     'n_outputs': n_outputs,
                     'n_steps' : n_steps,
                     'n_units' : n_units,
                     'n_hidden_neurons' : n_hidden_neurons,
                     'k_prob' : k_prob}
        
        self.w2v_model = w2v_model
        
        self._initialize_vars()
        self._initialize_weights()
        
        self.sess = tf.Session()
        if save_file:
            saver = tf.train.Saver()
            saver.restore(self.sess, save_file)
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        
        self.optimizer
        self.loss
        
    def _initialize_vars(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.params['n_steps'], self.params['n_inputs']))
        self.y_true = tf.placeholder(tf.int64, shape=(None))
        
        self.seq_length = tf.placeholder(tf.int64, shape=(None))
        self.reg_param = tf.placeholder(tf.float32,shape=())# lambda param
        
    def _initialize_weights(self):

        cell_factory = GRUCell(num_units=self.params['n_units'])
        cell_drop    = DropoutWrapper(cell_factory, self.params['k_prob'])
        __, states = tf.nn.dynamic_rnn(cell_drop, self.x, dtype=tf.float32,
                                        sequence_length=self.seq_length)
        
        hidden = fc(states, self.params['n_hidden_neurons'])
        self.output = fc(hidden, self.params['n_outputs'], activation_fn=None)
    
    @classmethod
    def load(cls, filename):
        
        with open(cls.PATH+filename+'_params'+cls.DATAEXT, 'r') as f:
            params = json.load(f)
        save_file = cls.PATH + filename + cls.TF_EXT
        return cls(**params, save_file=save_file)
    
    def save(self, filename='testing_tf_save'):
        saver = tf.train.Saver()
        path = self.__class__.PATH + filename
        save_file = path + self.__class__.TF_EXT
        saver.save(self.sess, save_file)
        with open(path+'_params'+self.__class__.DATAEXT,'w') as f: 
            json.dump(self.params, f)
        print("Trained Model Saved to '{}'.".format(save_file))
    
    def fit(self, rnn_training_data, 
                  reg_param=0.2, 
                  n_epochs = 40, 
                  batch_size=10, 
                  learn_rate=0.05):
        print("----")
        for tf_var in tf.trainable_variables():
            print(tf_var)
        print("----")
        print([v for v in tf.trainable_variables() if "bias" in v.name])
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

        correct = tf.nn.in_top_k(self.output, self.y_true, 3)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        train = rnn_training_data.training
        test  = rnn_training_data.test

        for epoch in range(n_epochs):
            
            for X_b, length_b, y_b in rnn_minibatches(train.X, train.length, train.Y, batch_size):
                train_dict = {self.x:X_b, 
                              self.seq_length: length_b, 
                              self.y_true: y_b, 
                              self.reg_param: reg_param}
                self.sess.run(self.optimizer, train_dict)
            if epoch % 2 == 0:
                full_train_dict = {self.x:train.X, 
                                   self.seq_length:train.length, 
                                   self.y_true:train.Y,
                                   self.reg_param: reg_param}
                test_dict = {self.x:test.X, 
                             self.seq_length:test.length, 
                             self.y_true:test.Y,
                             self.reg_param: reg_param}
                training_acc = self.sess.run(accuracy, full_train_dict)
                test_acc = self.sess.run(accuracy, test_dict)
                cur_loss = self.sess.run(self.loss, full_train_dict)
                print ("Epoch: ", epoch, "Training acc: ", 
                       training_acc*100, "Test acc: ", test_acc*100,
                       "Loss: ", cur_loss)    
    
    def predict_sentence(self, sentence):
        args = (sentence, self.w2v_model, self.params['n_steps'])
        x_data = RNNTrainingData.build_rnn_input( *args )
        return self.predict(x_data, len(sentence))
    
    def predict(self, x_data, seq_length=None):
        if not seq_length:
            seq_length = x_data.shape[-1]
        shape = reshape(-1, self.params['n_steps'], self.params['n_inputs'])
        pred = self.sess.run(self.output, {self.x:x_data.reshape(*shape), 
                                           self.seq_length:seq_length})
        return pred
    
    @property_with_check
    def loss(self):
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
        xentropy = softmax(labels=self.y_true, logits=self.output)
        base_loss = tf.reduce_mean(xentropy)
        
        l2_loss = self.reg_param * sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if not ("bias" in tf_var.name)
        )
        cost = base_loss + l2_loss
        cost = base_loss
        return cost
    
    @property_with_check
    def optimizer(self):
        opt = tf.train.GradientDescentOptimizer(0.05)
        opt = opt.minimize(self.loss)
        return opt