import numpy as np
import tensorflow.compat.v1 as tf
from math import ceil, sqrt

from ops import linear, non_linear

import logging
import pdb

def mlp(opts, input, output_dim, nlayers, init=None, stddev=0.0099999, bias=0., nonlinear='leaky_relu', eta1=1., eta2=1., scope=None, reuse=False):
    layer_x = input
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(nlayers-1):
            layer_x = linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                        output_dim=2, init=init,
                        stddev=bias, bias=bias,
                        scope='hid{}/lin'.format(i),
                        reuse=reuse)
            layer_x = non_linear(layer_x, nonlinear, eta1, eta2)
    # output layer
    outputs = linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim=output_dim, init=init,
                stddev=bias, bias=bias,
                scope='hid_lin_final',
                reuse=reuse)
    layer_x = non_linear(layer_x, nonlinear, eta1, eta2)

    return layer_x
