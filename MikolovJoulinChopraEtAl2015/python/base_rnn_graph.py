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
import math
import numpy as np
import tensorflow as tf

# Local imports
from batch_generator import batch_generator
from log_prob import log_prob

# Define base RNN TensorFlow graph class
class base_rnn_graph(object):
    
    # Place holder for graph constructor
    def __init__(self):
        print('TensorFlow graph not defined')
        
    # Placeholder function for cell definition
    def _cell(self):
        print('Cell not defined')
    
    # Placeholder function to set up cell parameters
    def _setup_cell_parameters(self):
        print('Cell parameters not defined')
        
    # Placeholder function to implement a tower to run part of a batch of training data on a GPU
    def _training_tower(self, i, tower, gpu):
        print('Training tower not defined')
        
    # Placeholder function to implement a tower to run part of a batch of validation data on a GPU
    def _validation_tower(self, tower, gpu):
        print('Validation tower not defined')
            
    # Train model parameters
    def train(self, learning_rate, learning_decay, num_epochs, summary_frequency, training_text, validation_text):

        # Generate training batches
        print('Training Batch Generator:')
        training_batches = []
        for tower in range(self._num_towers):
            training_batches.append(batch_generator(tower, training_text[tower], self._batch_size,
                                                    self._num_training_unfoldings, self._vocabulary_size))
        
        # Generate validation batches
        print('Validation Batch Generator:')
        validation_batches = []
        tower = 0
        for tower in range(self._num_towers):
            validation_batches.append(batch_generator(tower, validation_text[tower], 1,
                                                      self._num_validation_unfoldings, self._vocabulary_size))
        
        # Training loop
        batch_ctr = 0
        epoch_ctr = 0
        training_feed_dict = dict()
        validation_feed_dict = dict()
        with tf.Session(graph=self._graph, config=tf.ConfigProto(log_device_placement=True)) as session:
        
            session.run(self._initialization)
            print('Initialized')

            # Iterate over fixed number of training epochs
            for epoch in range(num_epochs):

                # Display the learning rate for this epoch
                print('Epoch: %d  Learning Rate: %.2f' % (epoch+1, learning_rate))

                # Training Step:

                # Iterate over training batches
                for tower in range(self._num_towers):
                    training_batches[tower].reset_token_idx()
                session.run(self._reset_training_state)
                for batch in range(training_batches[0].num_batches()):

                    # Get next training batch
                    training_batches_next = []
                    tower = 0
                    for tower in range(self._num_towers):
                        training_batches_next.append([])
                        training_batches_next[tower] = training_batches[tower].next()
                    batch_ctr += 1

                    # Optimization
                    training_feed_dict[self._learning_rate] = learning_rate
                    for tower in range(self._num_towers):
                        for i in range(self._num_training_unfoldings + 1):
                            training_feed_dict[self._training_data[tower][i]] = training_batches_next[tower][i]
                    session.run(self._optimize, feed_dict=training_feed_dict)

                    # Summarize current performance
                    if (batch+1) % summary_frequency == 0:
                        cst = session.run(self._cost, feed_dict=training_feed_dict)
                        print('     Total Batches: %d  Current Batch: %d  Cost: %.2f' % 
                              (batch_ctr, batch+1, cst))
                      
                # Validation Step:
        
                # Iterate over validation batches
                for tower in range(self._num_towers):
                    validation_batches[tower].reset_token_idx()
                session.run(self._reset_validation_state)
                validation_log_prob_sum = 0
                for _ in range(validation_batches[0].num_batches()):
                    
                    # Get next validation batch
                    validation_batches_next = []
                    tower = 0
                    for tower in range(self._num_towers):
                        validation_batches_next.append([])
                        validation_batches_next[tower] = validation_batches[tower].next()
                    
                    # Validation
                    validation_batches_next_label = []
                    for tower in range(self._num_towers):
                        validation_batches_next_label_tmp = []
                        for i in range(self._num_validation_unfoldings):
                            validation_feed_dict[self._validation_input[tower][i]] = validation_batches_next[tower][i]
                            validation_batches_next_label_tmp.append(validation_batches_next[tower][i+1])
                        validation_batches_next_label.append(validation_batches_next_label_tmp)
                    validation_prediction = session.run(self._validation_prediction, feed_dict=validation_feed_dict)
                    
                    # Summarize current performance
                    for tower in range(self._num_towers):
                        for i in range(self._num_validation_unfoldings):
                            validation_log_prob_sum = validation_log_prob_sum + \
                                log_prob(validation_prediction[tower][i], validation_batches_next_label[tower][i])
                    
                # Calculation validation perplexity
                N = self._num_towers*self._num_validation_unfoldings*validation_batches[0].num_batches()
                perplexity = float(2 ** (-validation_log_prob_sum / N))
                print('Epoch: %d  Validation Set Perplexity: %.2f' % (epoch+1, perplexity))

                # Update learning rate
                if epoch > 0 and perplexity > perplexity_last_epoch:
                    learning_rate *= learning_decay
                perplexity_last_epoch = perplexity