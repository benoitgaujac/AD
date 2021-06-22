import numpy as np
import tensorflow.compat.v1 as tf
from math import pi
import pdb

# wrapper to init weights
def custom_uniform(stdev, size):
    return np.random.uniform(low=-stdev * np.sqrt(3),
                            high=stdev * np.sqrt(3),
                            size=size
                            ).astype('float32')

def non_linear(inputs, type, eta1=1., eta2=1.):
    if type=='relu':
        return tf.nn.relu(inputs)
    elif type=='soft_plus':
        return tf.nn.softplus(inputs)
    elif type=='leaky_relu':
        return tf.nn.leaky_relu(inputs, eta1)
    elif type=='tanh':
        return eta2*tf.math.tanh(eta1*inputs)
    elif type=='sinh':
        return tf.math.sinh(inputs)
    elif type=='cubic':
        return inputs*inputs*inputs
    elif type=='linear':
        return eta1*inputs
    else:
        raise ValueError('Unknown {} non linear activtion' % type)
