# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This model is superceded by the Delta-RNN model given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 implemented in this repository using Python and Tensorflow.
#
# This code fails to implement hierarchical softmax at this time as Tensorflow does not appear to include an
# implementation.  Hierarchical softmax can be included at a future date when hierarchical softmax is available 
# for Tensorflow.
#
# Stuart Hagler, 2017

# Imports
import tensorflow as tf

# Local imports
from base_rnn_graph3 import base_rnn_graph3

# Define derived SCRN TensorFlow graph class
class scrn_graph(base_rnn_graph3):
    
    # Graph constructor
    def __init__(self, num_gpus, alpha, hidden_size, state_size, vocabulary_size, num_training_unfoldings,
                 num_validation_unfoldings, training_batch_size, validation_batch_size, optimization_frequency):
        
        # Input hyperparameters
        self._alpha = alpha
        
        base_rnn_graph3.__init__(self, num_gpus, hidden_size, state_size, vocabulary_size, num_training_unfoldings,
                                 num_validation_unfoldings, training_batch_size, validation_batch_size, optimization_frequency)
    
    # SCRN cell definition   .
    def _cell(self, x, h, s):
        with tf.name_scope('State'):
            state_arg = (1 - self._alpha) * tf.matmul(x, self._B) + self._alpha * s
            state = state_arg
        with tf.name_scope('Hidden'):
            hidden_arg = tf.matmul(s, self._P) + tf.matmul(x, self._A) + tf.matmul(h, self._R)
            hidden = tf.sigmoid(hidden_arg)
        with tf.name_scope('Output'):
            output_arg = tf.matmul(hidden, self._U) + tf.matmul(state, self._V) 
            output = output_arg
        return output, hidden, state
    
    # Setup SCRN cell parameters
    def _setup_cell_parameters(self):
        
        # Context embedding tensor.
        with tf.name_scope('B'):
            self._B = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._state_size], -0.1, 0.1))

        # Token embedding tensor.
        with tf.name_scope('A'):
            self._A = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._hidden_size], -0.1, 0.1))
            
        #
        with tf.name_scope('P'):
            self._P = tf.Variable(tf.truncated_normal([self._state_size, self._hidden_size], -0.1, 0.1))

        # Recurrent weights tensor and bias.
        with tf.name_scope('R'):
            self._R = tf.Variable(tf.truncated_normal([self._hidden_size, self._hidden_size], -0.1, 0.1))

        # Output update tensor and bias.
        with tf.name_scope('U'):
            self._U = tf.Variable(tf.truncated_normal([self._hidden_size, self._vocabulary_size], -0.1, 0.1))
        with tf.name_scope('V'):
            self._V = tf.Variable(tf.truncated_normal([self._state_size, self._vocabulary_size], -0.1, 0.1))