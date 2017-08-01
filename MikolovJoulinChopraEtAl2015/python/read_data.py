# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This model is superceded by the Delta-RNN model given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 implemented in this repository using Python and Tensorflow.
#
# A read data function that reads data in a zip-file for feeding into the LSTM, SCRN, and SRN models.
#
# Stuart Hagler, 2017

# usecase_flg = 1 for predicting letters
#               2 for predicting words with cutoff for infrequent words

# Imports
import tensorflow as tf
import zipfile

def read_data(usecase_flg, filename):
    # read datafile
    with zipfile.ZipFile(filename) as f:
        if usecase_flg == 1:
            raw_data = tf.compat.as_str(f.read(f.namelist()[0]))
        elif usecase_flg == 2:
            raw_data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    # Return data
    return raw_data