import numpy as np
import tensorflow.compat.v1 as tf

# import tensorflow as tf
# import tensorflow_probability as tfp
# from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU, Conv2D, Reshape
# from tensorflow.keras import Model
#
# from .case import Case
#
# from utils.train_utils import checkerboard
# tfd = tfp.distributions
# tfb = tfp.bijectors
# tfk = tf.keras
#
# tf.keras.backend.set_floatx('float32')
#
# print('tensorflow: ', tf.__version__)
# print('tensorflow-probability: ', tfp.__version__)

import ops._ops as _ops
from ops.normalization import Normalization
from ops.permutation import Permutation
import networks
import ops

import pdb

'''--------------------------------------------- mlp flow --------------------------------------------------------'''

def mlp(opts, inputs, scope=None, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = networks.mlp(opts, inputs, output_dim=2,
                                reuse=reuse)

    return outputs


'''--------------------------------------------- Planar Flow --------------------------------------------------------'''

def planar(opts, inputs, scope=None, reuse=False):
    '''
    Implementation of Planar Flow.
    '''

    def m(x):
        return -1 + tf.math.softplus(x)

    # init flow parameters
    with tf.variable_scope(scope, reuse=reuse):
        u = tf.get_variable('u', [2,], tf.float32, tf.random_uniform_initializer(minval=-1., maxval=1))
        w = tf.get_variable('w', [2,], tf.float32, tf.random_uniform_initializer(minval=-1., maxval=1))
        b = tf.get_variable('b', [1,], tf.float32, tf.random_uniform_initializer(minval=-1., maxval=1))

    # follow Appendix A.1. of Jimenez Rezende, D. and Mohamed, S.
    wu = tf.linalg.matmul(tf.expand_dims(w,0), tf.expand_dims(u,-1)) #[1,1]
    wu = tf.reshape(wu, [1,]) #[1,1]
    w_norm = w / tf.reduce_sum(tf.square(w))
    u_hat = u + (m(wu) - wu) * w_norm #[2,]
    u_hat = tf.reshape(u_hat, [1,2,1]) #[1,2,1]
    # flow
    outputs = tf.linalg.matmul(tf.reshape(w, [1,1,-1]), tf.expand_dims(inputs, -1)) + tf.reshape(b, [1,1,-1]) #[batch, 1,1]
    outputs = tf.expand_dims(inputs, axis=-1) + u_hat * _ops.non_linear(outputs, 'tanh') #[batch, 2,1]

    return tf.reshape(outputs, [-1,2])


'''--------------------------------------------- 2d glow flow --------------------------------------------------------'''

def glow(opts, inputs, scope=None, reuse=False, is_training=True):

    layer_x = inputs
    with tf.variable_scope(scope, reuse=reuse):
        # normalization
        layer_x = Normalization(opts, layer_x, scope, reuse, is_training)
        # flow perm
        layer_x = Permutation(opts, layer_x)
        # affine layer
        layer_x1, layer_x2 = tf.split(layer_x, 2, -1)
        h = networks.mlp(opts, layer_x2, output_dim=2*layer_x2.get_shape().as_list()[1],
                                    reuse=reuse)
        log_s, t = tf.split(h, 2, -1)
        # s = tf.nn.sigmoid(log_s)
        s = tf.math.exp(log_s)
        layer_x2 = s*layer_x2 + t
        outputs = tf.concat([layer_x1, layer_x2],axis=-1)

    return outputs


'''---------------------------------------------- main -----------------------------------------------'''

def flow(opts, inputs, scope=None, reuse=False, is_training=True):
    if opts['flow']=='identity':
        return inputs
    elif opts['flow']=='mlp':
        outputs = mlp(opts, inputs, scope, reuse)
    elif opts['flow']=='planar':
        outputs = planar(opts, inputs, scope, reuse)
    elif opts['flow']=='realNVP':
        outputs = realNVP(opts, inputs, scope, reuse, is_training)
    elif opts['flow']=='glow':
        outputs = glow(opts, inputs, scope, reuse, is_training)
    else:
        raise ValueError('Unknown {} flow' % flow_type)

    return outputs
