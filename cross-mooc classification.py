from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import cPickle,string
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten 
from keras.layers.embeddings import Embedding 
from keras.layers.convolutional import Conv1D, MaxPooling1D 
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
# set parameters: 
maxlen = 500 
batch_size = 32 
filters = 250 
kernel_size = 5 
hidden_dims = 250 
epochs = 15
nb_epoch_t = 50

