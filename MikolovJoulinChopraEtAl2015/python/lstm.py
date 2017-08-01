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
from base_rnn_graph2 import base_rnn_graph2
                
# Define derived LSTM TensorFlow graph class
class lstm_graph(base_rnn_graph2):
    
    # LSTM cell definition   .
    def _cell(self, x, h, c):
        forget_arg = tf.matmul(x, self._Wf) + tf.matmul(h, self._Uf)
        forget_gate = tf.sigmoid(forget_arg + self._forget_bias)
        input_arg = tf.matmul(x, self._Wi) + tf.matmul(h, self._Ui)
        input_gate = tf.sigmoid(input_arg + self._input_bias)
        output_arg = tf.matmul(x, self._Wo) + tf.matmul(h, self._Uo)
        output_gate = tf.sigmoid(output_arg + self._output_bias)
        update_arg = tf.matmul(x, self._Wc) + tf.matmul(h, self._Uc)
        state = forget_gate * c + input_gate * tf.tanh(update_arg + self._update_bias)
        output = output_gate * tf.tanh(state)
        return output, state
    
    # Setup LSTM cell parameters
    def _setup_cell_parameters(self):
        
        #
        if self._hidden_size != self._state_size:
            print("Hidden size must equal state size")
        
        # Forget gate input and output tensor and bias.
        self._Wf = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._hidden_size], -0.1, 0.1))
        self._Uf = tf.Variable(tf.truncated_normal([self._hidden_size, self._hidden_size], -0.1, 0.1))
        self._forget_bias = tf.Variable(tf.zeros([1, self._hidden_size]))

        # Input gate input and output tensor and bias.
        self._Wi = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._hidden_size], -0.1, 0.1))
        self._Ui = tf.Variable(tf.truncated_normal([self._hidden_size, self._hidden_size], -0.1, 0.1))
        self._input_bias = tf.Variable(tf.zeros([1, self._hidden_size]))

        # Output gate input and output tensor and bias.
        self._Wo = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._hidden_size], -0.1, 0.1))
        self._Uo = tf.Variable(tf.truncated_normal([self._hidden_size, self._hidden_size], -0.1, 0.1))
        self._output_bias = tf.Variable(tf.zeros([1, self._hidden_size]))

        # Cell state update input and output tensor and bias.
        self._Wc = tf.Variable(tf.truncated_normal([self._vocabulary_size, self._hidden_size], -0.1, 0.1))
        self._Uc = tf.Variable(tf.truncated_normal([self._hidden_size, self._hidden_size], -0.1, 0.1))
        self._update_bias = tf.Variable(tf.zeros([1, self._hidden_size]))

        # Softmax weight tensor and bias.
        self._W = tf.Variable(tf.truncated_normal([self._hidden_size, self._vocabulary_size], -0.1, 0.1))
        self._W_bias = tf.Variable(tf.zeros([self._vocabulary_size]))