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
from base_rnn_graph1 import base_rnn_graph1
                
# Define derived SRN TensorFlow graph class
class srn_graph(base_rnn_graph1):
    
    # SRN cell definition   .
    def _cell(self, x, h):
        with tf.name_scope('Hidden'):
            hidden_arg = tf.matmul(x, self._A) + tf.matmul(h, self._R)
            hidden = tf.sigmoid(hidden_arg)
        with tf.name_scope('Output'):
            output_arg = tf.matmul(hidden, self._U)
            output = output_arg
        return output, hidden
    
    # Setup SRN cell parameters
    def _setup_cell_parameters(self):
        
        # Token embedding tensor.
        with tf.name_scope('A'):
            self._A = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._hidden_size], -0.1, 0.1))

        # Recurrent weights tensor and bias.
        with tf.name_scope('R'):
            self._R = tf.Variable(tf.truncated_normal([self._hidden_size, self._hidden_size], -0.1, 0.1))

        # Output update tensor and bias.
        with tf.name_scope('U'):
            self._U = tf.Variable(tf.truncated_normal([self._hidden_size, self._vocabulary_size], -0.1, 0.1))