import numpy as np
import tensorflow.compat.v1 as tf

import pdb


### batchnorm
def Batchnorm_layers(opts, inputs, scope=None, is_training=False, reuse=None, scale=True, center=True, fused=False):
    """Batch normalization based on tf.layers.batch_normalization.

    """
    return tf.layers.batch_normalization(
        inputs, momentum=opts['batch_norm_momentum'],
        epsilon=opts['batch_norm_eps'],
        center=center, scale=scale,
        training=is_training, reuse=reuse,
        name=scope, fused=fused)


def Normalization(opts, inputs, scope=None, reuse=False, is_training=True):
    if opts['normalization'] == 'none':
        outputs = inputs
    elif opts['normalization'] == 'batchnorm':
        outputs = Batchnorm_layers(opts, inputs, scope=None, reuse=None, is_training=False)
    else:
        raise ValueError('{} norm not implemented yet' % opts['normalization'])

    return outputs
