import numpy as np
import tensorflow.compat.v1 as tf
from math import pi
import pdb


def init_w(opts, scope):
    init = opts['w_init']
    stddev = opts['init_std']
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
            raise Exception('Invalid %s mlp initialization!' % opts['mlp_init'])
        return W

def init_diagonal(opts, scope):
    init = opts['d_init']
    stddev = opts['init_std']
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
            eps = 10e-5
            D = tf.clip_by_value(D, -opts['clip_alpha_value'], opts['clip_alpha_value'])
            D = tf.concat([tf.exp(D),1./(eps + tf.exp(D))], axis=0)
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

def non_linear(inputs, type, alpha=1.):
    if type=='relu':
        return tf.nn.relu(inputs)
    elif type=='soft_plus':
        return tf.nn.softplus(inputs)
    elif type=='leaky_relu':
        return tf.nn.leaky_relu(inputs)
    elif type=='tanh':
        return tf.math.tanh(inputs)
    elif type=='sinh':
        return tf.math.sinh(inputs)
    elif type=='cubic':
        return inputs*inputs*inputs
    elif type=='linear':
        return alpha*inputs
    else:
        raise ValueError('Unknown {} non linear activtion' % type)
