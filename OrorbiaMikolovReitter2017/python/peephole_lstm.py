# Delta Recurrent Neural Network (Delta-RNN) Framework
#
# This gives an implementation of the Delta-RNN framework given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 using Python and Tensorflow.
#
# This code implements a variety of RNN models using the Delta-RNN Framework
#
# Stuart Hagler, 2017

# Imports
import tensorflow as tf

# Local imports
from delta_rnn import delta_rnn_graph

# Define derived Peephole LSTM TensorFlow graph class
class peephole_lstm_graph(delta_rnn_graph):
    
    # Graph constructor
    def __init__(self, num_gpus, alpha, c_size, h_size, vocabulary_size, num_training_unfoldings,
                 num_validation_unfoldings, training_batch_size, validation_batch_size, optimization_frequency):
        
        delta_rnn_graph.__init__(self, num_gpus, alpha, h_size, h_size, vocabulary_size, num_training_unfoldings,
                                 num_validation_unfoldings, training_batch_size, validation_batch_size, optimization_frequency)
    
    # Peephole LSTM cell definition   .
    def _cell(self, x, c, h):
        i = tf.sigmoid(tf.matmul(x, self._Wi) + tf.matmul(h, self._Vi) + self._bi)
        f = tf.sigmoid(tf.matmul(x, self._Wf) + tf.matmul(h, self._Vf) + self._bf)
        o = tf.sigmoid(tf.matmul(x, self._Wo) + tf.matmul(h, self._Vo) + self._bo)
        z = tf.sigmoid(tf.matmul(x, self._Wz) + tf.matmul(h, self._Vz) + self._bz)
        c = f*c + i*z
        h = o*tf.tanh(c)
        o = tf.nn.xw_plus_b(o, self._W, self._b)
        return o, c, h
    
    # Setup Peephole LSTM cell parameters
    def _setup_cell_parameters(self):
        self._Wi = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vi = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bi = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wf = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vf = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bf = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wo = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vo = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bo = tf.Variable(tf.zeros([1, self._h_size]))
        self._Wz = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
        self._Vz = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))
        self._bz = tf.Variable(tf.zeros([1, self._h_size]))
        self._W = tf.Variable(tf.truncated_normal([self._h_size, self._vocabulary_size], -0.1, 0.1))
        self._b = tf.Variable(tf.zeros([self._vocabulary_size]))