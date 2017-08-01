# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This model is superceded by the Delta-RNN model given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 implemented in this repository using Python and Tensorflow.
#
# The log probability function used to calculate the validation perplexity the LSTM, SCRN, and SRN models.
#
# Stuart Hagler, 2017

# Imports
from itertools import compress
import numpy as np

# Calculate the log-probability of the label given predictions
def log_prob(predictions, label):
    predictions[predictions < 1e-10] = 1e-10
    label_probability = [ predictions[i] for i in [ label > 0 ] ]
    return np.log2(label_probability)