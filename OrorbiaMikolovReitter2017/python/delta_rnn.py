# Delta Recurrent Neural Network (Delta-RNN) Framework
#
# This gives an implementation of the Delta-RNN framework given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 using Python and Tensorflow.
#
# This code implements a variety of RNN models using the Delta-RNN Framework
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

# Tensorflow graph
class delta_rnn_graph(base_rnn_graph):
    
    # Graph constructor
    def __init__(self, num_gpus, alpha, c_size, h_size, vocabulary_size, num_training_unfoldings,
                 num_validation_unfoldings, training_batch_size, validation_batch_size, optimization_frequency):
        
        # Input hyperparameters
        self._alpha = alpha
        self._training_batch_size = training_batch_size
        self._c_size = c_size
        self._h_size = h_size
        self._num_gpus = num_gpus
        self._num_training_unfoldings = num_training_unfoldings
        self._num_validation_unfoldings = num_validation_unfoldings
        self._optimization_frequency = optimization_frequency
        self._training_batch_size = training_batch_size
        self._validation_batch_size = validation_batch_size
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
            self._training_c_saved = []
            self._training_h_saved = []
            for _ in range(self._num_towers):
                training_data_tmp = []
                for _ in range(num_training_unfoldings + 1):
                    training_data_tmp.append(tf.placeholder(tf.float32, shape=[self._training_batch_size,
                                                                               self._vocabulary_size]))
                self._training_data.append(training_data_tmp)
                self._training_c_saved.append(tf.Variable(tf.zeros([self._training_batch_size, self._c_size]),
                                                          trainable=False))
                self._training_h_saved.append(tf.Variable(tf.zeros([self._training_batch_size, self._h_size]),
                                                          trainable=False))
                
            # Validation data
            self._validation_input = []
            self._validation_c_saved = []
            self._validation_h_saved = []
            for _ in range(self._num_towers):
                validation_input_tmp = []
                for _ in range(num_validation_unfoldings):
                    validation_input_tmp.append(tf.placeholder(tf.float32, shape=[self._validation_batch_size,
                                                                                  self._vocabulary_size]))
                self._validation_input.append(validation_input_tmp)
                self._validation_c_saved.append(tf.Variable(tf.zeros([self._validation_batch_size, self._c_size]),
                                                            trainable=False))
                self._validation_h_saved.append(tf.Variable(tf.zeros([self._validation_batch_size, self._h_size]),
                                                            trainable=False))
                
            # Optimizer hyperparameters
            self._clip_norm = tf.placeholder(tf.float32)
            self._learning_rate = tf.placeholder(tf.float32)
            self._momentum = tf.placeholder(tf.float32)
                
            # Optimizer
            self._optimizer = tf.train.MomentumOptimizer(self._learning_rate, self._momentum)
                    
            # Training:
            
            # Reset training state
            self._reset_training_state = \
                [ tf.group(self._training_c_saved[tower].assign(tf.zeros([self._training_batch_size, self._c_size])),
                           self._training_h_saved[tower].assign(tf.zeros([self._training_batch_size, self._h_size]))) \
                  for tower in range(self._num_towers) ]
            
            # Train cell on training data
            for i in range(self._num_training_unfoldings // self._optimization_frequency):
                training_labels = []
                training_outputs = []
                for tower in range(self._num_towers):
                    training_labels.append([])
                    training_outputs.append([])
                for tower in range(self._num_towers):
                    training_outputs[tower], training_labels[tower] = \
                        self._training_tower(i, tower, tower)
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
                
            # Summarize training performance
            tf.summary.scalar('cost', self._cost)
            self._training_summary = tf.summary.merge_all()
                
            # Initialization:
            
            self._initialization = tf.global_variables_initializer()
                
            # Validation:
    
            # Reset validation state
            self._reset_validation_state = \
                [ tf.group(self._validation_c_saved[tower].assign(tf.zeros([self._validation_batch_size, self._c_size])),
                           self._validation_h_saved[tower].assign(tf.zeros([self._validation_batch_size, self._h_size]))) \
                  for tower in range(self._num_towers) ]

            # Run cell on validation data
            validation_outputs = []
            for tower in range(self._num_towers):
                validation_outputs.append([])
            for tower in range(self._num_towers):
                validation_outputs[tower] = self._validation_tower(tower, tower)
            logits = validation_outputs

            # Validation prediction, replace with hierarchical softmax in the future
            self._validation_prediction = tf.nn.softmax(logits)
    
    # Implements a tower to run part of a batch of training data on a GPU
    def _training_tower(self, i, tower, gpu):
        
        with tf.device("/gpu:%d" % gpu):
        
            # Get saved training state
            c = self._training_c_saved[tower]
            h = self._training_h_saved[tower]

            # Run training data through LSTM cells
            labels = []
            outputs = []
            for j in range(self._optimization_frequency):
                x = self._training_data[tower][i*self._optimization_frequency + j]
                label = self._training_data[tower][i*self._optimization_frequency + j + 1]
                o, c, h = self._cell(x, c, h)
                labels.append(label)
                outputs.append(o)

            # Save training state and return training outputs
            with tf.control_dependencies([self._training_c_saved[tower].assign(c),
                                          self._training_h_saved[tower].assign(h)]):
                return outputs, labels
        
    # Implements a tower to run part of a batch of validation data on a GPU
    def _validation_tower(self, tower, gpu):
        
        with tf.device("/gpu:%d" % gpu):
        
            # Get saved validation state
            c = self._validation_c_saved[tower]
            h = self._validation_h_saved[tower]

            # Run validation data through LSTM cells
            outputs = []
            for i in range(self._num_validation_unfoldings):
                x = self._validation_input[tower][i]
                o, c, h = self._cell(x, c, h)
                outputs.append(o)

            # Save validation state and return validation outputs
            with tf.control_dependencies([self._validation_c_saved[tower].assign(c), 
                                          self._validation_h_saved[tower].assign(h)]):
                return outputs