import numpy as np
import tensorflow.compat.v1 as tf
from math import pi

import ops
# import networks

import pdb

class Model(object):

    def __init__(self, opts):
        self.opts = opts

    def init_model_params(self, opts, reuse=False):
        """
        init all the different model parameters
        """
        with tf.variable_scope('score', reuse=reuse):
            W = ops.init_w(self.opts, 'W')
            if self.opts['train_d']:
                d = ops.init_diagonal(self.opts, 'D')
            else:
                d = tf.constant([0, 100],dtype='float32')
            D = tf.diag(d)
            phi = ops.init_rotation(self.opts, 'phi')
            phi = tf.clip_by_value(phi, 0., pi)
            rot = tf.stack([tf.math.cos(phi), -tf.math.sin(phi),
                            tf.math.sin(phi), tf.math.cos(phi)], 0)
            V = tf.reshape(rot, [2,2])
        return W, d, D, phi, V

    def score(self, inputs, reuse=False):
        """
        Return the output of last layer of the score function defined by:
        output = W * a(V^-1DV * inputs)
        inputs:  [batch,2]
        outputs: [batch,]
        """

        # get model params
        W, _, D, _, V = self.init_model_params(self.opts, reuse=reuse)
        # affine transform
        A = tf.linalg.matmul(D, tf.transpose(V))
        A = tf.linalg.matmul(V, A)
        # score fct
        score = tf.linalg.matmul(tf.expand_dims(A, 0), tf.expand_dims(inputs, -1))
        score = ops.non_linear(score, self.opts['score_non_linear'])
        if self.opts['train_w']:
            score = tf.linalg.matmul(tf.expand_dims(W, 0), score)
        else:
            score = tf.reduce_sum(score, axis=1)
        if self.opts['clip_score']:
            score = tf.clip_by_value(score, -self.opts['clip_score_value'],
                                self.opts['clip_score_value'])

        return tf.reshape(score, [-1,1])

    def D_reg(self, reuse=False):
        """
        Return dilatation regulation
        """
        _, _, D, _, _ = self.init_model_params(self.opts, reuse=reuse)
        mask = tf.Variable([1., 0.])
        if self.opts['d_reg']=='trace':
            reg = tf.linalg.trace(D)
        elif self.opts['d_reg']=='frob':
            reg = tf.sqrt(tf.linalg.trace(tf.square(D)))
        elif self.opts['d_reg']=='det':
            reg = tf.linalg.det(D)
        elif self.opts['d_reg']=='alpha':
            # pdb.set_trace()
            diag = tf.linalg.diag_part(D)
            reg = tf.abs(tf.abs(diag*mask) - self.opts['d_reg_value'])
            reg = tf.reduce_sum(reg)
        else:
            raise ValueError('Unknown {} D reg.' % self.opts['d_reg'])
        if self.opts['clip_d_reg']:
            reg = tf.clip_by_value(reg, -self.opts['clip_d_reg_value'],
                                    self.opts['clip_d_reg_value'])

        return reg


    def W_reg(self, reuse=False):
        """
        Return W regulation
        """
        W, _, _, _, _ = self.init_model_params(self.opts, reuse=reuse)
        if self.opts['w_reg']=='l1':
            reg = tf.reduce_sum(tf.math.abs(W))
        elif self.opts['w_reg']=='l2':
            reg = tf.reduce_sum(tf.math.square(W))
        elif self.opts['w_reg']=='l2sq':
            reg = tf.reduce_sum(tf.math.square(W))
        else:
            raise ValueError('Unknown {} W reg.' % self.opts['w_reg'])
        if self.opts['clip_w_reg']:
            reg = tf.clip_by_value(reg, -self.opts['clip_w_reg_value'],
                                    self.opts['clip_w_reg_value'])

        return reg


class Affine(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def losses(self, inputs, reuse=False):
        """
        return score, D reg and W reg for the affine transformation
        """
        score = self.score(inputs, reuse=reuse)
        d_reg = self.D_reg(reuse=True)
        w_reg = self.W_reg(reuse=True)

        return score, d_reg, w_reg


class NonAffine(Model):

    def __init__(self, opts):
        super().__init__(opts)

    def losses(self, inputs, reuse=False):
        """
        return score, D reg and W reg for the affine transformation
        """

        raise ValueError('Non affine score fct not implemented')

        # score = networks.nn(self.opts, inputs, reuse=reuse)
        # score = self.score(score, reuse=reuse)
        # d_reg = self.D_reg(reuse=reuse)
        # w_reg = self.W_reg(reuse=reuse)
        #
        # return tf.math.abs(score), d_reg, w_reg
