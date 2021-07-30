import numpy as np
import tensorflow.compat.v1 as tf
from math import pi
import normalizing_flow

import ops._ops as _ops
import ops.init_affine as init_affine
import networks

import pdb

class Model(object):

    def __init__(self, opts):
        self.opts = opts

    def init_model_params(self, opts, reuse=False):
        """
        init all the different model parameters
        """
        with tf.variable_scope('score', reuse=reuse):
            W = init_affine.init_w(self.opts)
            if self.opts['train_d']:
                d = init_affine.init_d(self.opts)
            else:
                d = tf.constant([0, 100],dtype='float32')
            D = tf.diag(d)
            phi = init_affine.init_phi(self.opts)
            # phi = tf.clip_by_value(phi, 0., pi)
            rot = tf.stack([tf.math.cos(phi), -tf.math.sin(phi),
                            tf.math.sin(phi), tf.math.cos(phi)], 0)
            V = tf.reshape(rot, [2,2])
        return W, d, D, phi, V


    def flow(self, inputs, reuse=False, is_training=True):
        """
        Perform normalizing flow on the inputs
        """

        outputs = []
        layer_x = inputs
        with tf.variable_scope('flow', reuse=reuse):
            for i in range(self.opts['nsteps']):
                layer_x = normalizing_flow.flow(self.opts, layer_x, 'flow_step{}'.format(i), reuse, is_training)
                outputs.append(layer_x)

        return outputs


    def score(self, inputs, reuse=False, is_training=True):
        """
        Return the output of last layer of the score function defined by:
        output = W * a(V^-1DV * inputs)
        inputs:  [batch,2]
        outputs: [batch,]
        """

        ### normalizing flow
        inputs = self.flow(inputs, reuse, is_training)
        ### affine transform
        # get model params
        W, _, D, _, V = self.init_model_params(self.opts, reuse=reuse)
        # rotate if needed
        if self.opts['rotate']:
            A = tf.linalg.matmul(D, tf.transpose(V))
            A = tf.linalg.matmul(V, A)
        else:
            A = D
        # score fct
        score = tf.linalg.matmul(tf.expand_dims(A, 0), tf.expand_dims(inputs[-1], -1))
        score = _ops.non_linear(score, self.opts['score_nonlinear'])
        if self.opts['train_w']:
            score = tf.linalg.matmul(tf.expand_dims(W, 0), score)
        else:
            score = tf.sqrt(tf.reduce_sum(tf.square(score), axis=1))
        if self.opts['clip_score']:
            score = tf.clip_by_value(score, -self.opts['clip_score_value'],
                                self.opts['clip_score_value'])

        return tf.reshape(score, [-1,1]), inputs

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


    def losses(self, inputs, reuse=False, is_training=True):
        """
        return score, transformed outputs, D & W reg
        """
        score, _ = self.score(inputs, reuse=reuse, is_training=is_training)
        d_reg = self.D_reg(reuse=True)
        w_reg = self.W_reg(reuse=True)

        return score, d_reg, w_reg
