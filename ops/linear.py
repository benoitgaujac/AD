import numpy as np
import tensorflow.compat.v1 as tf
from math import pi
from ops._ops import custom_uniform as custom_uniform

import pdb

def Linear(opts, input, input_dim, output_dim, init=None, stddev=0.0099999, bias=0., scope=None, reuse=None):
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
