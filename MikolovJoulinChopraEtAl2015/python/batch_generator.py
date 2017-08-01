# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This model is superceded by the Delta-RNN model given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 implemented in this repository using Python and Tensorflow.
#
# The batch generator class that is used to feed the data to the LSTM, SCRN, and SRN models.
#
# Stuart Hagler, 2017

# Imports
import numpy as np

#
class batch_generator(object):
    
    #
    def __init__(self, tower, text, batch_size, num_unfoldings, vocabulary_size):
        
        #
        self._batch_size = batch_size
        self._num_unfoldings = num_unfoldings
        self._vocabulary_size = vocabulary_size
        
        #
        self._effective_batch_size = self._batch_size * self._num_unfoldings
        self._dropped_text_size = len(text) % self._effective_batch_size
        self._text_size = len(text) - self._dropped_text_size
        self._text = text[:self._text_size]
        self._text_size = len(self._text)
        
        self._sub_text_size = self._text_size // self._batch_size
        self._num_batches = self._sub_text_size // self._num_unfoldings
        self._offsets = [ i * self._sub_text_size for i in range(self._batch_size) ]
        self._cursor = [ offset * self._sub_text_size for offset in range(batch_size)]
        
        #
        self._token_idx = 0
        
        #
        self._last_batch = self._next_batch()
        
        #
        self._print_summary(tower, text)
    
    #
    def _next_batch(self):
        
        # Generate a batch starting with token at self._first_token_idx
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for i in range(self._batch_size):
            batch[i, self._text[self._offsets[i] + self._token_idx]] = 1.0
        self._token_idx += 1
        return batch
    
    #
    def _print_summary(self, tower, text):
        print('     Tower: %d' % tower)
        print('          Input Text Size: %d' % len(text))
        print('          Cut Text Size: %d' % self._text_size)
        print('          Subtext Size: %d' % self._sub_text_size)
        print('          Dropped Text Size: %d' % self._dropped_text_size)
        print('          Effective Batch Size: %d' % self._effective_batch_size)
        print('          Number of Batches: %d' % self._num_batches)
        
    #
    def next(self):

        #
        batches = [self._last_batch]
        for step in range(self._num_unfoldings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches
    
    #
    def num_batches(self):
        return self._num_batches
    
    #
    def reset_token_idx(self):
        self._token_idx = 0