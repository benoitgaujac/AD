import numpy as np
import tensorflow.compat.v1 as tf
from math import pi
from ops._ops import custom_uniform as custom_uniform

import pdb


def init_w(opts):
    init = opts['w_init']
    stddev = opts['w_init_std']
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

def init_d(opts):
    init = opts['d_init']
    stddev = opts['d_init_std']
    const = opts['d_const']
    if const:
        shape = [1,]
    else:
        shape = [2,]
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

def init_phi(opts):
    phi = tf.get_variable( 'phi', [1,], tf.float32,
        tf.random_uniform_initializer(minval=0.,maxval=pi))
    return phi
