import numpy as np
import tensorflow.compat.v1 as tf
import math

import ops
# import networks

import pdb

class Model(object):

    def __init__(self, opts):
        self.opts = opts
        # define different weights of the last layer
        self.W = ops.init_w(self.opts, 'score')
        self.D = ops.init_diagonal(self.opts, 'score')
        self.V = ops.init_rotation(self.opts, 'score')

    def score(self, inputs, reuse=False):
        """
        Return the output of last layer of the score function defined by:
        output = W * a(V^-1DV * inputs)
        inputs:  [batch,2]
        outputs: [batch,]
        """
        with tf.variable_scope('score', reuse=reuse):
            # affine transform
            A = tf.linalg.matmul(self.D, self.V)
            A = tf.linalg.matmul(tf.transpose(self.V), A)
            # score fct
            score = tf.linalg.matmul(tf.expand_dims(A, 0), tf.expand_dims(inputs, -1))
            score = ops.non_linear(score, self.opts['score_non_linear'],
                                    self.opts['clip_score_value'])
            if self.opts['clip_score']:
                score = tf.clip_by_value(score, -self.opts['clip_score_value'],
                                    self.opts['clip_score_value'])
            if self.opts['learned_proj']:
                score = tf.linalg.matmul(tf.expand_dims(self.W, 0), score)
            else:
                score = tf.reduce_sum(score, axis=1)

        return tf.reshape(score, [-1,1])

    def D_reg(self, reuse=False):
        """
        Return dilatation regulation
        """
        with tf.variable_scope('score', reuse=reuse):
            if self.opts['d_reg']=='trace':
                reg = tf.linalg.trace(self.D)
            elif self.opts['d_reg']=='frob':
                reg = tf.sqrt(tf.linalg.trace(tf.square(self.D)))
            elif self.opts['d_reg']=='det':
                reg = tf.linalg.det(self.D)
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
        with tf.variable_scope('score', reuse=reuse):
            if self.opts['w_reg']=='l1':
                reg = tf.reduce_sum(tf.math.abs(self.W))
            elif self.opts['w_reg']=='l2':
                reg = tf.reduce_sum(tf.math.square(self.W))
            elif self.opts['w_reg']=='l2sq':
                reg = tf.reduce_sum(tf.math.square(self.W))
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
        d_reg = self.D_reg(reuse=reuse)
        w_reg = self.W_reg(reuse=reuse)

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
