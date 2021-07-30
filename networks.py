import numpy as np
import tensorflow.compat.v1 as tf
from math import ceil, sqrt

import ops._ops as _ops
import ops.linear as linear

import logging
import pdb

def mlp(opts, input, output_dim, reuse=False):
    layer_x = input
    for i in range(opts['mlp_nlayers']-1):
        layer_x = linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                    output_dim=opts['mlp_nunits'],
                    init=opts['mlp_init'],
                    stddev=opts['mlp_init_std'],
                    bias=opts['mlp_init_bias'],
                    scope='hid{}/lin'.format(i),
                    reuse=reuse)
        layer_x = _ops.non_linear(layer_x, opts['mlp_nonlinear'], 0.2)
    # output layer
    outputs = linear.Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                output_dim=output_dim,
                init=opts['mlp_init_final'],
                stddev=opts['mlp_init_std_final'],
                bias=opts['mlp_init_bias_final'],
                scope='hid_lin_final',
                reuse=reuse)
    outputs = _ops.non_linear(outputs, opts['mlp_nonlinear_final'],
                opts['mlp_eta1'],
                opts['mlp_eta2'])

    return outputs
