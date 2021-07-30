import numpy as np
import tensorflow.compat.v1 as tf

import pdb


def Permutation(opts, inputs, scope=None, reuse=False, is_training=True):
    if opts['permutation'] == 'reverse':
        outputs = inputs[:,::-1]
    else:
        raise ValueError('{} permutation not implemented yet' % opts['permutation'])

    return outputs
