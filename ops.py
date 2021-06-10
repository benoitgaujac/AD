import numpy as np
import tensorflow.compat.v1 as tf
from math import pi
import pdb


def init_w(opts, scope):
    init = opts['w_init']
    stddev = opts['w_init_std']
    with tf.variable_scope(scope):
        if init == 'normal':
            W = tf.get_variable(
                'W', [1,2], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        elif init == 'glorot':
            weight_values = custom_uniform(
                np.sqrt(2./3.), (1, 2))
            W = tf.get_variable(
                'W', [1,2], tf.float32, weight_values)
        elif init == 'he':
            weight_values = custom_uniform(
                np.sqrt(2.), (1, 2))
            W = tf.get_variable(
                'W', [1,2], tf.float32, weight_values)
        elif init == 'glorot_he':
            weight_values = custom_uniform(
                np.sqrt(4./3.), (1, 2))
            W = tf.get_variable(
                'W', [1,2], tf.float32, weight_values)
        elif init == 'glorot_uniform':
            W = tf.get_variable(
                'W', [1, 2], tf.float32,
                tf.glorot_uniform_initializer())
        elif init[0] == 'uniform':
            W = tf.get_variable(
                'W', [1, 2], tf.float32,
                tf.random_uniform_initializer(minval=-1., maxval=1))
        else:
            raise Exception('Invalid %s mlp initialization!' % init)
        return W

def init_diagonal(opts, scope):
    init = opts['d_init']
    stddev = opts['d_init_std']
    const = opts['d_const']
    if const:
        shape = [1,]
    else:
        shape = [2,]
    with tf.variable_scope(scope):
        if init == 'normal':
            D = tf.get_variable(
                'D', shape, tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        elif init == 'glorot':
            weight_values = custom_uniform(
                np.sqrt(2./3.), (1, 2))
            D = tf.get_variable(
                'D', shape, tf.float32, weight_values)
        elif init == 'he':
            weight_values = custom_uniform(
                np.sqrt(2.), (1, 2))
            D = tf.get_variable(
                'D', shape, tf.float32, weight_values)
        elif init == 'glorot_he':
            weight_values = custom_uniform(
                np.sqrt(4./3.), (1, 2))
            D = tf.get_variable(
                'D', shape, tf.float32, weight_values)
        elif init == 'glorot_uniform':
            D = tf.get_variable(
                'D', shape, tf.float32,
                tf.glorot_uniform_initializer())
        elif init[0] == 'uniform':
            D = tf.get_variable(
                'D', shape, tf.float32,
                tf.random_uniform_initializer(minval=-1.,maxval=1))
        else:
            raise Exception('Unknown %s initialization for D' % opts['d_init'])
        if const:
            # eps = 10e-5
            # D = tf.concat([tf.exp(D),1./(eps + tf.exp(D))], axis=0)
            D = tf.concat([D,1./D], axis=0)
        return D

def init_rotation(opts, scope):
    with tf.variable_scope(scope):
        phi = tf.get_variable( 'phi', [1,], tf.float32,
            tf.random_uniform_initializer(minval=0.,maxval=pi))
    return phi

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

def linear(opts, input, input_dim, output_dim, init=None, stddev=0.0099999, bias=0., scope=None, reuse=None):
    """Fully connected linear layer.

    Args:
        input: [num_points, ...] tensor, where every point can have an
            arbitrary shape. In case points are more than 1 dimensional,
            we will stretch them out in [numpoints, prod(dims)].
        output_dim: number of features for the output. I.e., the second
            dimensionality of the matrix W.
    """

    shape = input.get_shape().as_list()
    assert len(shape) > 0
    if len(shape) > 2:
        # This means points contained in input have more than one
        # dimensions. In this case we first stretch them in one
        # dimensional vectors
        input = tf.reshape(input, [-1, input_dim])

    with tf.variable_scope(scope or "lin", reuse=reuse):
        if init == 'normal' or init == None:
            W = tf.get_variable(
                "W", [input_dim, output_dim], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        elif init == 'glorot':
            weight_values = custom_uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim))
            W = tf.get_variable(
                "W", initializer=weight_values, dtype=tf.float32)
        elif init == 'he':
            weight_values = custom_uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim))
            W = tf.get_variable(
                "W", initializer=weight_values, dtype=tf.float32)
        elif init == 'glorot_he':
            weight_values = custom_uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim))
            W = tf.get_variable(
                "W", initializer=weight_values, dtype=tf.float32)
        elif init == 'glorot_uniform':
            W = tf.get_variable(
                "W", [input_dim, output_dim], tf.float32,
                tf.glorot_uniform_initializer())
        elif init == 'uniform':
            W = tf.get_variable(
                "W", [input_dim, output_dim], tf.float32,
                tf.random_uniform_initializer(
                    minval=-initialization[1],
                    maxval=initialization[1]))
        else:
            raise Exception('Invalid %s mlp initialization!' % init)
        b = tf.get_variable(
            "b", [output_dim],
            initializer=tf.constant_initializer(bias))


    return tf.matmul(input, W) + bias
