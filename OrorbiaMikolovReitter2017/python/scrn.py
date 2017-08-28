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

# Define derived SCRN TensorFlow graph class
class scrn_graph(delta_rnn_graph):
    
    # SCRN cell definition   .
    def _cell(self, x, c, h):
        with tf.name_scope('Hidden'):
            h = tf.sigmoid(tf.matmul(c, self._P) + tf.matmul(x, self._A) + tf.matmul(h, self._R))
        with tf.name_scope('Cell'):
            c = (1 - self._alpha) * tf.matmul(x, self._B) + self._alpha * c
        with tf.name_scope('Output'):
            o = tf.matmul(h, self._U) + tf.matmul(c, self._V) 
        return o, c, h
    
    # Setup SCRN cell parameters
    def _setup_cell_parameters(self):
        
        # Context embedding tensor.
        with tf.name_scope('B'):
            self._B = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._c_size], -0.1, 0.1))

        # Token embedding tensor.
        with tf.name_scope('A'):
            self._A = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._h_size], -0.1, 0.1))
            
        #
        with tf.name_scope('P'):
            self._P = tf.Variable(tf.truncated_normal([self._c_size, self._h_size], -0.1, 0.1))

        # Recurrent weights tensor and bias.
        with tf.name_scope('R'):
            self._R = tf.Variable(tf.truncated_normal([self._h_size, self._h_size], -0.1, 0.1))

        # Output update tensor and bias.
        with tf.name_scope('U'):
            self._U = tf.Variable(tf.truncated_normal([self._h_size, self._vocabulary_size], -0.1, 0.1))
        with tf.name_scope('V'):
            self._V = tf.Variable(tf.truncated_normal([self._c_size, self._vocabulary_size], -0.1, 0.1))