
"""
Train Ad score function
"""
import os
import logging

import numpy as np
import tensorflow.compat.v1 as tf
from math import pi

import utils
from plots import plot_train, plot_transformation, plot_score_heatmap
import models

import pdb

class Run(object):

    def __init__(self, opts, data):

        logging.error('Building the Tensorflow Graph')
        self.opts = opts

        # - Data
        self.data = data

        # - Placeholders
        self.add_ph()

        # - Instantiate Model
        self.model = models.Model(self.opts)

        # - Data/label
        x, y = self.data.next_element
        y = tf.reshape(y, [-1,1])

        # - Define Objectives
        score, self.d_reg, self.w_reg = self.model.losses(
                                    inputs=x,
                                    reuse=False,
                                    is_training=self.is_training)
        # obj
        self.score = tf.reduce_mean(score)
        # self.objective = tf.math.abs(score) - tf.cast(y, dtype=tf.float32) * (self.lmbda*self.d_reg + self.gamma*self.w_reg)
        # self.objective = score - tf.cast(y, dtype=tf.float32) * (self.lmbda*self.d_reg + self.gamma*self.w_reg)
        # self.objective = tf.math.abs(score) - self.lmbda*self.d_reg - self.gamma*self.w_reg
        # self.objective = score - self.lmbda*self.d_reg - self.gamma*self.w_reg
        # self.objective = tf.cast(y, dtype=tf.float32) * tf.math.abs(score) \
        #                     + (1. - tf.cast(y, dtype=tf.float32)) * score \
        #                     - self.lmbda*self.d_reg + self.gamma*self.w_reg
        self.objective = tf.cast(y, dtype=tf.float32) * (tf.math.abs(score) - self.lmbda*self.d_reg - self.gamma*self.w_reg) \
                            - (1. - tf.cast(y, dtype=tf.float32)) * (tf.math.abs(score) + self.lmbda*self.d_reg + self.gamma*self.w_reg)
        self.objective = tf.reduce_mean(self.objective)

        # - Optimizers, savers, etc
        self.add_optimizers()

        # - nominal/anomalous score
        score, _ = self.model.score(inputs=self.x, reuse=True, is_training=False)
        score_anomalies = self.y*score + (1-self.y)*tf.abs(score)
        self.score_anomalies = tf.reduce_mean(score_anomalies)
        # self.heatmap_score_anomalies = tf.math.abs(score)
        self.heatmap_score_anomalies = score

        # - transformed outputs for non affine model
        _, self.transformed = self.model.score(inputs=self.x, reuse=True, is_training=False)

        # - Params values
        _, self.d, _, self.phi, _ = self.model.init_model_params(self.opts, reuse=True)

        # - Init iterators, sess, saver and load trained weights if needed, else init variables
        self.sess = tf.Session()
        self.train_handle, self.test_handle = self.data.init_iterator(self.sess)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.initializer = tf.global_variables_initializer()
        self.sess.graph.finalize()


    def add_ph(self):
        self.lr_decay = tf.placeholder(tf.float32, [], name='rate_decay_ph')
        self.x = tf.placeholder(tf.float32, [None,2], name='points')
        self.y = tf.placeholder(tf.float32, [None,], name='labels')
        self.gamma = tf.placeholder(tf.float32, [], name='gamma_ph')
        self.lmbda = tf.placeholder(tf.float32, [], name='lmbda_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')


    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if self.opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif self.opts['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(lr,
                                    beta1=self.opts['adam_beta1'],
                                    beta2=self.opts['adam_beta2'])
        else:
            raise ValueError('Unknown {} optimizer' % self.opts['optimizer'])


    def add_optimizers(self):
        lr = self.opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        vars = []
        vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='score/phi')
        vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='flow')
        if self.opts['train_w']:
            vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='score/W')
        if self.opts['train_d']:
            vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='score/D')
        self.opt = opt.minimize(loss=self.objective, var_list=vars)


    def train(self, WEIGHTS_FILE=None):
        logging.error('\nTraining {} flow'.format(self.opts['flow']))
        exp_dir = self.opts['exp_dir']

        # - Set up for training
        train_size = self.data.train_size
        batches_num = self.data.train_size//self.opts['batch_size']
        logging.error('\nTrain size: {}, trBatch num.: {}, Ite. num: {}'.format(
                                    train_size,
                                    batches_num,
                                    self.opts['it_num']))

        # - Init all monitoring variables
        Losses, Losses_test = [], []
        Scores_anomalies = []
        Psi, D = [], []

        # - Init decay lr and gamma
        decay = 1.
        decay_rate = 0.9
        fix_decay_steps = 25000

        # - Testing iterations number
        test_it_num = self.data.test_size // self.opts['batch_size']

        # - Load trained model or init variables
        if self.opts['use_trained']:
            if WEIGHTS_FILE is None:
                    raise Exception("No model/weights provided")
            else:
                if not tf.gfile.IsDirectory(self.opts['exp_dir']):
                    raise Exception("model doesn't exist")
                WEIGHTS_PATH = os.path.join(self.opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
                if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                    raise Exception("weights file doesn't exist")
                self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.initializer)

        # - Training
        for it in range(self.opts['it_num']):
            # Saver
            if it > 0 and it % self.opts['save_every'] == 0:
                self.saver.save(self.sess,
                                    os.path.join(exp_dir, 'checkpoints', 'trained-wae'),
                                    global_step=it)
            #####  TRAINING LOOP #####
            it += 1
            # training
            _ = self.sess.run(self.opt, feed_dict={
                                    self.data.handle: self.train_handle,
                                    self.lr_decay: decay,
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda'],
                                    self.is_training: True})

            ##### TESTING LOOP #####
            if it % self.opts['evaluate_every'] == 0:
                # training loss
                feed_dict={self.data.handle: self.train_handle,
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda'],
                                    self.is_training: False}
                losses = self.sess.run([self.objective,
                                    self.score,
                                    self.d_reg,
                                    self.w_reg],
                                    feed_dict=feed_dict)
                Losses.append(losses)
                # model params
                psi, d = self.sess.run([self.phi, self.d], feed_dict={})
                Psi.append(psi[0]%pi)
                D.append(d)
                # testing loss
                losses, scores_anomalies = np.zeros(4), np.zeros(2)
                for it_ in range(test_it_num):
                    # testing losses
                    feed_dict={self.data.handle: self.test_handle,
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda'],
                                    self.is_training: False}
                    loss = self.sess.run([self.objective,
                                    self.score,
                                    self.d_reg,
                                    self.w_reg],
                                    feed_dict=feed_dict)
                    losses += np.array(loss) / test_it_num
                    # nominal score
                    batch_inputs = self.data._sample_observation(
                                    self.opts['batch_size'],
                                    self.opts['dataset'],
                                    True)
                    feed_dict={self.x: batch_inputs[0],
                                    self.y: batch_inputs[1],
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda']}
                    score_nominal = self.sess.run(self.score_anomalies,
                                    feed_dict=feed_dict)
                    # anomalous score
                    batch_inputs = self.data._sample_observation(
                                    self.opts['batch_size'],
                                    self.opts['dataset'],
                                    False)
                    feed_dict={self.x: batch_inputs[0],
                                    self.y: batch_inputs[1],
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda']}
                    score_anomalous = self.sess.run(self.score_anomalies,
                                    feed_dict=feed_dict)
                    scores_anomalies += np.array((score_nominal, score_anomalous)) / test_it_num
                Losses_test.append(losses)
                Scores_anomalies.append(scores_anomalies)

                # Printing various loss values
                logging.error('')
                debug_str = 'Ite.: %d/%d, ' % (it, self.opts['it_num'])
                logging.error(debug_str)
                debug_str = 'trLoss=%.3f, teLoss=%.3f' % (
                                    Losses[-1][0], Losses_test[-1][0])
                logging.error(debug_str)
                debug_str = 'teScore=%.3f, teDilate=%.3f, teW=%.3f' % (
                                    Losses_test[-1][1],
                                    self.opts['lmbda']*Losses_test[-1][2],
                                    self.opts['gamma']*Losses_test[-1][3])
                logging.error(debug_str)
                debug_str = 'trScore=%.3f, trDilate=%.3f, trW=%.3f' % (
                                    Losses[-1][1],
                                    self.opts['lmbda']*Losses[-1][2],
                                    self.opts['gamma']*Losses[-1][3])
                logging.error(debug_str)
                debug_str = 'NoScore=%.3f, AnScore=%.3f' % (
                                    Scores_anomalies[-1][0],
                                    Scores_anomalies[-1][1])
                logging.error(debug_str)

            if it % self.opts['plot_every'] == 0:
                # non affine transformation
                batch_inputs = self.data._sample_observation(
                                500,
                                self.opts['dataset'],
                                True)
                if self.opts['flow']!='identity':
                    feed_dict={self.x: batch_inputs[0],
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda']}
                    transformed = self.sess.run(self.transformed,
                                    feed_dict=feed_dict)
                    if self.opts['plot_trans']:
                        plot_transformation(batch_inputs[0], transformed,
                                        self.opts['exp_dir'],
                                        'train_plots',
                                        'it%07d' % (it))
                else:
                    transformed = None
                # score fct heatmap
                xs = np.linspace(-1, 1, 101, endpoint=True)
                ys = np.linspace(-1, 1, 101, endpoint=True)
                xv, yv = np.meshgrid(xs,ys)
                grid = np.stack((xv,yv),axis=-1)
                grid = grid.reshape([-1,2])
                feed_dict={self.x: grid,
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda']}
                heatmap = self.sess.run(self.heatmap_score_anomalies,
                                    feed_dict=feed_dict)
                heatmap = heatmap.reshape([101,101])[:,::-1]
                if self.opts['plot_hm']:
                    pdb.set_trace()
                    plot_score_heatmap(heatmap, self.opts['exp_dir'],
                                        'train_plots',
                                        'it%07d' % (it))

                # plot
                plot_train(self.opts, Losses, Losses_test,
                                    Scores_anomalies, heatmap,
                                    batch_inputs[0], transformed,
                                    Psi, D, exp_dir,
                                    'res_it%07d.png' % (it))

            # - Update learning rate if necessary and it
            if self.opts['lr_decay']:
                # decaying every fix_decay_steps
                if it % fix_decay_steps == 0:
                    decay = decay_rate ** (int(it / fix_decay_steps))
                    logging.error('Reduction in lr: %f\n' % decay)

        # - Save the final model
        if self.opts['save_final'] and it > 0:
            self.saver.save(self.sess, os.path.join(exp_dir,
                                                'checkpoints',
                                                'trained-{}-final'.format(self.opts['flow'])),
                                                global_step=it)

        # - Finale losses & scores
        # training loss
        feed_dict={self.data.handle: self.train_handle,
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda'],
                                    self.is_training: False}
        losses = self.sess.run([self.objective,
                                    self.score,
                                    self.d_reg,
                                    self.w_reg],
                            feed_dict=feed_dict)
        Losses.append(losses)
        # testing loss
        losses, scores_anomalies = np.zeros(4), np.zeros(2)
        for it_ in range(test_it_num):
            # testing losses
            feed_dict={self.data.handle: self.test_handle,
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda'],
                                    self.is_training: False}
            loss = self.sess.run([self.objective,
                                    self.score,
                                    self.d_reg,
                                    self.w_reg],
                                    feed_dict=feed_dict)
            losses += np.array(loss) / test_it_num
            # nominal score
            batch_inputs = self.data._sample_observation(
                                    self.opts['batch_size'],
                                    self.opts['dataset'],
                                    True)
            feed_dict={self.x: batch_inputs[0],
                                    self.y: batch_inputs[1],
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda']}
            score_nominal = self.sess.run(self.score_anomalies,
                                    feed_dict=feed_dict)
            # anomalous score
            batch_inputs = self.data._sample_observation(
                                    self.opts['batch_size'],
                                    self.opts['dataset'],
                                    False)
            feed_dict={self.x: batch_inputs[0],
                                    self.y: batch_inputs[1],
                                    self.gamma: self.opts['gamma'],
                                    self.lmbda: self.opts['lmbda']}
            score_anomalous = self.sess.run(self.score_anomalies,
                                    feed_dict=feed_dict)
            scores_anomalies += np.array((score_nominal, score_anomalous)) / test_it_num
        Losses_test.append(losses)
        Scores_anomalies.append(scores_anomalies)

        # Printing various loss values
        logging.error('')
        debug_str = 'Ite.: %d/%d, ' % (it, self.opts['it_num'])
        logging.error(debug_str)
        debug_str = 'trLoss=%.3f, teLoss=%.3f' % (
                                    Losses[-1][0], Losses_test[-1][0])
        logging.error(debug_str)
        debug_str = 'teScore=%.3f, teDilate=%.3f, teW=%.3f' % (
                                    Losses_test[-1][1],
                                    self.opts['lmbda']*Losses_test[-1][2],
                                    self.opts['gamma']*Losses_test[-1][3])
        logging.error(debug_str)
        debug_str = 'trScore=%.3f, trDilate=%.3f, trW=%.3f' % (
                                    Losses[-1][1],
                                    self.opts['lmbda']*Losses[-1][2],
                                    self.opts['gamma']*Losses[-1][3])
        logging.error(debug_str)
        debug_str = 'NoScore=%.3f, AnScore=%.3f' % (
                                    Scores_anomalies[-1][0],
                                    Scores_anomalies[-1][1])
        logging.error(debug_str)

        # -- save training data
        if self.opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path, name),
                    loss=np.array(Losses),
                    loss_test=np.array(Losses_test),
                    loss_anomalies=np.array(Scores_anomalies),
                    theta=self.opts['theta'], coef=np.array(self.opts['coef']),
                    psi=np.array(Psi), d = np.array(D))


    # def plot(self, WEIGHTS_FILE=None):
    def plot(self, exp_dir, WEIGHTS_FILE=None):
        """
        Plots transformed input and score heatmap
        """
        # - exp_dir
        self.opts['exp_dir'] = exp_dir

        # - Load trained model
        if WEIGHTS_FILE is None:
                raise Exception("No model/weights provided")
        else:
            WEIGHTS_PATH = os.path.join(self.opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
            if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_PATH)

        # - transformed inputs
        batch_inputs = self.data._sample_observation(
                        500,
                        self.opts['dataset'],
                        True)
        if self.opts['flow']!='identity':
            feed_dict={self.x: batch_inputs[0],
                            self.gamma: self.opts['gamma'],
                            self.lmbda: self.opts['lmbda']}
            transformed = self.sess.run(self.transformed,
                            feed_dict=feed_dict)
            plot_transformation(batch_inputs[0], transformed, self.opts['exp_dir'], 'test_plots', self.opts['dataset'])

        # - score heatmap
        xs = np.linspace(-1, 1, 101, endpoint=True)
        ys = np.linspace(-1, 1, 101, endpoint=True)
        xv, yv = np.meshgrid(xs,ys)
        grid = np.stack((xv,yv),axis=-1)
        grid = grid.reshape([-1,2])
        feed_dict={self.x: grid,
                            self.gamma: self.opts['gamma'],
                            self.lmbda: self.opts['lmbda']}
        heatmap = self.sess.run(self.heatmap_score_anomalies,
                            feed_dict=feed_dict)
        heatmap = heatmap.reshape([101,101])[:,::-1]
        plot_score_heatmap(heatmap, self.opts['exp_dir'], 'test_plots', self.opts['dataset'])

    # def test(self, MODEL_PATH=None, WEIGHTS_FILE=None):
    #     """
    #     Test model and save different metrics
    #     """
    #
    #     opts = self.opts
    #
    #     # - Load trained weights
    #     if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
    #         raise Exception("weights file doesn't exist")
    #     self.saver.restore(self.sess, WEIGHTS_PATH)
    #
    #     # - Set up
    #     test_size = data.test_size
    #     batch_size_te = min(test_size,1000)
    #     batches_num_te = int(test_size/batch_size_te)+1
    #     # - Init all monitoring variables
    #     Loss, Loss_rec, MSE = 0., 0., 0.
    #     Divergences = []
    #     MIG, factorVAE, SAP = 0., 0., 0.
    #     real_blurr, blurr, fid_scores = 0., 0., 0.
    #     if opts['true_gen_model']:
    #         codes, codes_mean = np.zeros((batches_num_te*batch_size_te,opts['zdim'])), np.zeros((batches_num_te*batch_size_te,opts['zdim']))
    #         labels = np.zeros((batches_num_te*batch_size_te,len(data.factor_indices)))
    #     # - Testing loop
    #     for it_ in range(batches_num_te):
    #         # Sample batches of data points
    #         data_ids = np.random.choice(test_size, batch_size_te, replace=True)
    #         batch_images_test = data.get_batch_img(data_ids, 'test').astype(np.float32)
    #         batch_pz_samples_test = sample_pz(opts, self.pz_params, batch_size_te)
    #         test_feed_dict = {self.batch: batch_images_test,
    #                           self.samples_pz: batch_pz_samples_test,
    #                           self.obj_fn_coeffs: opts['obj_fn_coeffs'],
    #                           self.is_training: False}
    #         [loss, l_rec, mse, divergences, z, z_mean, samples] = self.sess.run([self.objective,
    #                                          self.loss_reconstruct,
    #                                          self.mse,
    #                                          self.divergences,
    #                                          self.z_samples,
    #                                          self.z_mean,
    #                                          self.generated_x],
    #                                         feed_dict=test_feed_dict)
    #         Loss += loss / batches_num_te
    #         Loss_rec += l_rec / batches_num_te
    #         MSE += mse / batches_num_te
    #         if len(Divergences)>0:
    #             Divergences[-1] += np.array(divergences) / batches_num_te
    #         else:
    #             Divergences.append(np.array(divergences) / batches_num_te)
    #         # storing labels and factors
    #         if opts['true_gen_model']:
    #                 codes[batch_size_te*it_:batch_size_te*(it_+1)] = z
    #                 codes_mean[batch_size_te*it_:batch_size_te*(it_+1)] = z_mean
    #                 labels[batch_size_te*it_:batch_size_te*(it_+1)] = data.get_batch_label(data_ids,'test')[:,data.factor_indices]
    #         # fid score
    #         if opts['fid']:
    #             # Load inception mean samples for train set
    #             trained_stats = os.path.join(inception_path, 'fid_stats.npz')
    #             # Load trained stats
    #             f = np.load(trained_stats)
    #             self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
    #             f.close()
    #             # Compute bluriness of real data
    #             real_blurriness = self.sess.run(self.blurriness,
    #                                         feed_dict={ self.batch: batch_images_test})
    #             real_blurr += np.mean(real_blurriness) / batches_num_te
    #             # Compute gen blur
    #             gen_blurr = self.sess.run(self.blurriness,
    #                                         feed_dict={self.batch: samples})
    #             blurr += np.mean(gen_blurr) / batches_num_te
    #             # Compute FID score
    #             # First convert to RGB
    #             if np.shape(samples)[-1] == 1:
    #                 # We have greyscale
    #                 samples = self.sess.run(tf.image.grayscale_to_rgb(samples))
    #             preds_incep = self.inception_sess.run(self.inception_layer,
    #                           feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
    #             preds_incep = preds_incep.reshape((batch_size_te,-1))
    #             mu_gen = np.mean(preds_incep, axis=0)
    #             sigma_gen = np.cov(preds_incep, rowvar=False)
    #             fid_score = fid.calculate_frechet_distance(mu_gen, sigma_gen,
    #                                         self.mu_train,
    #                                         self.sigma_train,
    #                                         eps=1e-6)
    #             fid_scores += fid_score / batches_num_te
    #     # - Compute disentanglment metrics
    #     if opts['true_gen_model']:
    #         MIG.append(self.compute_mig(codes_mean, labels))
    #         factorVAE.append(self.compute_factorVAE(data, codes))
    #         SAP.append(self.compute_SAP(data))
    #
    #     # - Printing various loss values
    #     if verbose=='high':
    #         debug_str = 'Testing done.'
    #         logging.error(debug_str)
    #         if opts['true_gen_model']:
    #             debug_str = 'MIG=%.3f, factorVAE=%.3f, SAP=%.3f' % (
    #                                         MIG,
    #                                         factorVAE,
    #                                         SAP)
    #             logging.error(debug_str)
    #         if opts['fid']:
    #             debug_str = 'Real blurr=%10.3e, blurr=%10.3e, FID=%.3f \n ' % (
    #                                         real_blurr,
    #                                         blurr,
    #                                         fid_scores)
    #             logging.error(debug_str)
    #
    #         if opts['model'] == 'BetaVAE':
    #             debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, gamma*KL=%10.3e \n '  % (
    #                                         Loss,
    #                                         Loss_rec,
    #                                         MSE,
    #                                         Divergences)
    #             logging.error(debug_str)
    #         elif opts['model'] == 'BetaTCVAE':
    #             debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*TC=%10.3e, KL=%10.3e \n '  % (
    #                                         Loss,
    #                                         Loss_rec,
    #                                         MSE,
    #                                         Divergences[0],
    #                                         Divergences[1])
    #             logging.error(debug_str)
    #         elif opts['model'] == 'FactorVAE':
    #             debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*KL=%10.3e, g*TC=%10.3e, \n '  % (
    #                                         Loss,
    #                                         Loss_rec,
    #                                         MSE,
    #                                         Divergences[0],
    #                                         Divergences[1])
    #             logging.error(debug_str)
    #         elif opts['model'] == 'WAE':
    #             debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*MMD=%10.3e \n ' % (
    #                                         Loss,
    #                                         Loss_rec,
    #                                         MSE,
    #                                         Divergences)
    #             logging.error(debug_str)
    #         elif opts['model'] == 'disWAE':
    #             debug_str = 'LOSS=%.3f, REC=%.3f, MSE=%.3f, b*HSIC=%10.3e, g*DIMWISE=%10.3e, WAE=%10.3e' % (
    #                                         Loss,
    #                                         Loss_rec,
    #                                         MSE,
    #                                         Divergences[0],
    #                                         Divergences[1],
    #                                         Divergences[2])
    #             logging.error(debug_str)
    #         elif opts['model'] == 'TCWAE_MWS' or opts['model'] == 'TCWAE_GAN':
    #             debug_str = 'LOSS=%.3f, REC=%.3f,l1*TC=%10.3e, MSE=%.3f, l2*DIMWISE=%10.3e, WAE=%10.3e' % (
    #                                         Loss,
    #                                         Loss_rec,
    #                                         MSE,
    #                                         Divergences[0],
    #                                         Divergences[1],
    #                                         Divergences[2])
    #             logging.error(debug_str)
    #         else:
    #             raise NotImplementedError('Model type not recognised')
    #
    #
    #     # - save testing data
    #     data_dir = 'test_data'
    #     save_path = os.path.join(opts['exp_dir'], data_dir)
    #     utils.create_dir(save_path)
    #     name = 'res_test_final'
    #     np.savez(os.path.join(save_path, name),
    #             loss=np.array(Loss),
    #             loss_rec=np.array(Loss_rec),
    #             mse = np.array(MSE),
    #             divergences=Divergences,
    #             mig=np.array(MIG),
    #             factorVAE=np.array(factorVAE),
    #             sap=np.array(SAP),
    #             real_blurr=np.array(real_blurr),
    #             blurr=np.array(blurr),
    #             fid=np.array(fid_scores))
