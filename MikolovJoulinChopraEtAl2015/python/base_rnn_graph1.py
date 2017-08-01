# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SRN model for comparison with the SCRN model given in Mikolov et al. 2015,
# arXiv:1412.7753 [cs.NE], https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
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
import math
import numpy as np
import tensorflow as tf

# Local imports
from base_rnn_graph import base_rnn_graph
from batch_generator import batch_generator
from log_prob import log_prob
    
# Define derived RNN TensorFlow graph class with saved hidden vectors
class base_rnn_graph1(base_rnn_graph):
    
    # Graph constructor
    def __init__(self, num_gpus, hidden_size, state_size, vocabulary_size, num_training_unfoldings,
                 num_validation_unfoldings, batch_size, optimization_frequency, clip_norm, momentum):
        
        # Input hyperparameters
        self._batch_size = batch_size
        self._clip_norm = clip_norm
        self._hidden_size = hidden_size
        self._momentum = momentum
        self._num_gpus = num_gpus
        self._num_training_unfoldings = num_training_unfoldings
        self._num_validation_unfoldings = num_validation_unfoldings
        self._optimization_frequency = optimization_frequency
        self._vocabulary_size = vocabulary_size
        
        # Derived hyperparameters
        self._num_towers = self._num_gpus
        
        # Graph definition
        self._graph = tf.Graph()
        with self._graph.as_default():

            # Cell parameter definitions
            self._setup_cell_parameters()
            
            # Training data
            self._training_data = []
            self._training_hidden_saved = []
            for _ in range(self._num_towers):
                training_data_tmp = []
                for _ in range(num_training_unfoldings + 1):
                    training_data_tmp.append(tf.placeholder(tf.float32, shape=[self._batch_size, self._vocabulary_size]))
                self._training_data.append(training_data_tmp)
                self._training_hidden_saved.append(tf.Variable(tf.zeros([self._batch_size, self._hidden_size]),
                                                               trainable=False))
            
            # Validation data
            self._validation_input = []
            self._validation_hidden_saved = []
            for _ in range(self._num_towers):
                validation_input_tmp = []
                for _ in range(num_validation_unfoldings):
                    validation_input_tmp.append(tf.placeholder(tf.float32, shape=[1, self._vocabulary_size]))
                self._validation_input.append(validation_input_tmp)
                self._validation_hidden_saved.append(tf.Variable(tf.zeros([1, self._hidden_size]), trainable=False))
                
            # Optimizer hyperparameters
            self._learning_rate = tf.placeholder(tf.float32)
                
            # Optimizer
            self._optimizer = tf.train.MomentumOptimizer(self._learning_rate, self._momentum)
                    
            # Training:
            
            # Reset training state
            self._reset_training_state = \
                [ tf.group(self._training_hidden_saved[tower].assign(tf.zeros([batch_size, hidden_size]))) \
                  for tower in range(self._num_towers) ]
            
            # Train RNN on training data
            for i in range(self._num_training_unfoldings // self._optimization_frequency):
                training_labels = []
                training_outputs = []
                for tower in range(self._num_towers):
                    training_labels.append([])
                    training_outputs.append([])
                for tower in range(self._num_towers):
                    training_outputs[tower], training_labels[tower] = self._training_tower(i, tower, tower)
                all_training_outputs = []
                all_training_labels = []
                for tower in range(self._num_towers):
                    all_training_outputs += training_outputs[tower]
                    all_training_labels += training_labels[tower]
                logits = tf.concat(all_training_outputs, 0)
                labels = tf.concat(all_training_labels, 0)

                # Replace with hierarchical softmax in the future
                self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

                gradients, variables = zip(*self._optimizer.compute_gradients(self._cost))
                gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
                self._optimize = self._optimizer.apply_gradients(zip(gradients, variables))
                
            # Initialization:
            
            self._initialization = tf.global_variables_initializer()
                    
            # Validation:
    
            # Reset validation state
            self._reset_validation_state = \
                [ tf.group(self._validation_hidden_saved[tower].assign(tf.zeros([1, hidden_size]))) \
                  for tower in range(self._num_towers) ]

            # Run RNN on validation data
            validation_outputs = []
            for tower in range(self._num_towers):
                validation_outputs.append([])
            for tower in range(self._num_towers):
                validation_outputs[tower] = self._validation_tower(tower,tower)
            logits = validation_outputs

            # Validation prediction, replace with hierarchical softmax in the future
            self._validation_prediction = tf.nn.softmax(logits)