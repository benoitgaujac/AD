import os
from datetime import datetime
import logging
import argparse
from math import pi, sqrt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np

import configs
from train import Run
from datahandler import DataHandler
import utils


import pdb

parser = argparse.ArgumentParser()
# run setup
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--dataset", default='line',
                    help='dataset')
parser.add_argument("--anomalous", action='store_true', default=False,
                    help='whether or not use anomalous data')
parser.add_argument("--fixed_dataset", action='store_true', default=False,
                    help='whether or not to sample randomly nominal manifold')
parser.add_argument("--num_it", type=int, default=10000,
                    help='iteration number')
parser.add_argument("--batch_size", type=int, default=100,
                    help='batch size')
parser.add_argument("--lr", type=float, default=0.001,
                    help='learning rate size')
# path setup
parser.add_argument("--res_dir", type=str,
                    help='root directory for res')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='dir in which experiment outputs are saved')
# exp setup
parser.add_argument("--exp_id", type=int,
                    help='exp. id')
# model set up
parser.add_argument("--flow", default='identity',
                    help='score flow to use')
parser.add_argument("--scr_nonlin", default='linear',
                    help='non linear activation for score fct')
parser.add_argument("--train_w", action='store_true', default=False,
                    help='whether to learn linear proj')
parser.add_argument("--gamma", type=float, default=0.,
                    help='weight regulation')
parser.add_argument("--train_d", action='store_false', default=True,
                    help='whether to learn D')
parser.add_argument("--d_const", action='store_false', default=True,
                    help='whether to set beta=1/alpha')
parser.add_argument("--lmbda", type=float, default=0.,
                    help='dilatation reg')
# saving opt
parser.add_argument('--save_model', action='store_false', default=True,
                    help='save final model weights [True/False]')
parser.add_argument("--save_data", action='store_false', default=True,
                    help='save training data')
# weights
parser.add_argument("--weights_file")


FLAGS = parser.parse_args()


def main():

    # Select dataset to use
    if FLAGS.dataset == 'line':
        opts = configs.config_line
        if FLAGS.fixed_dataset:
            configs.config_line['fixed_dataset'] = True
            configs.config_line['coef'] = [sqrt(2)*0.75]
            configs.config_line['theta'] = pi/4.
    elif FLAGS.dataset == 'quadratic':
        opts = configs.config_quadratic
        if FLAGS.fixed_dataset:
            configs.config_quadratic['fixed_dataset'] = True
            configs.config_quadratic['coef'] = [1., 0., -0.5]
            configs.config_quadratic['theta'] = -pi/2.
    elif FLAGS.dataset == 'cubic':
        opts = configs.config_cubic
        if FLAGS.fixed_dataset:
            configs.config_cubic['fixed_dataset'] = True
            configs.config_cubic['coef'] = [10., 0., 0., 0.]
            configs.config_cubic['theta'] = -pi/2.
    else:
        raise ValueError('Unknown {} dataset' % FLAGS.dataset)

    if FLAGS.anomalous:
        opts['use_anomalous'] = FLAGS.anomalous

    # Exp setup
    """    # multi configs of training w, D, constrained D
    exp = list(itertools.product([False,],
                                [False,],
                                [False,],
                                [0. ,0., 0., 0.]
                                ))
    exp += list(itertools.product([False,],
                                [True,],
                                [False, True],
                                [0. ,0., 0., 0.]
                                ))
    exp_id = (FLAGS.exp_id-1) % len(exp)
    opts['train_w'] = exp[exp_id][0]
    opts['train_d'] = exp[exp_id][1]
    opts['d_const'] = exp[exp_id][2]
    opts['lmbda'] = exp[exp_id][3]
    if opts['train_w']:
        opts['gamma'] = FLAGS.gamma
    else:
        opts['gamma'] = 0.
    """
    """ # Different alpha reg and lambda combination
    exp = list(itertools.product([0.1, 1., 10.],
                                [0.1, 1., 10.]))
    # setting exp id
    exp_id = (FLAGS.exp_id-1) % len(exp)
    opts['lmbda'] = exp[exp_id][0]
    opts['d_reg_value'] =  exp[exp_id][1]
    opts['train_w'] = FLAGS.train_w
    if opts['train_w']:
        opts['gamma'] = FLAGS.gamma
    else:
        opts['gamma'] = 0.
    opts['train_d'] = FLAGS.train_d
    opts['d_const'] = FLAGS.d_const
    """
    # Different scaling factor for non affine model
    exp = [1,2,3,4]
    # setting exp id
    exp_id = (FLAGS.exp_id-1) % len(exp)
    opts['nsteps'] = exp[exp_id]

    # Model set up
    opts['flow'] = FLAGS.flow
    opts['score_nonlinear'] = FLAGS.scr_nonlin
    opts['train_w'] = FLAGS.train_w
    if opts['train_w']:
        opts['gamma'] = FLAGS.gamma
    else:
        opts['gamma'] = 0.
    opts['train_d'] = FLAGS.train_d
    if opts['train_d']:
        opts['lmbda'] = FLAGS.lmbda
    else:
        opts['lmbda'] = 0.
    opts['d_const'] = FLAGS.d_const

    # Create directories
    if FLAGS.res_dir:
            results_dir = FLAGS.res_dir
    else:
        results_dir = 'results'
    if not tf.io.gfile.isdir(results_dir):
        utils.create_dir(results_dir)
    data_dir = os.path.join(results_dir, opts['dataset'])
    if not tf.io.gfile.isdir(data_dir):
        utils.create_dir(data_dir)
    out_dir = os.path.join(data_dir, FLAGS.out_dir)
    if not tf.io.gfile.isdir(out_dir):
        utils.create_dir(out_dir)
    opts['out_dir'] = os.path.join(out_dir, 'nsteps{}'.format(opts['nsteps']))
    if not tf.io.gfile.isdir(opts['out_dir']):
        utils.create_dir(opts['out_dir'])
    if FLAGS.res_dir:
        exp_name = FLAGS.res_dir + '_'
    else:
        exp_name = ''
    exp_name += 'w_' + str(opts['train_w']) + '_d_' + str(opts['train_d'])
    exp_name += '_constr_d_' + str(opts['d_const'])
    exp_name += '_g' + str(opts['gamma']) + '_l' + str(opts['lmbda'])
    exp_name += '_run_' + str(FLAGS.exp_id)
    opts['exp_dir'] = os.path.join(opts['out_dir'], exp_name)
    opts['exp_dir'] = os.path.join(opts['out_dir'],
                        '{}_{:%Y_%m_%d_%H_%M}'.format(
                        exp_name, datetime.now()))

    if not tf.io.gfile.isdir(opts['exp_dir']):
        utils.create_dir(opts['exp_dir'])
        utils.create_dir(os.path.join(opts['exp_dir'], 'checkpoints'))

    # Verbose
    logging.basicConfig(filename=os.path.join(opts['exp_dir'],'outputs.log'),
                        level=logging.INFO, format='%(asctime)s - %(message)s')

    opts['it_num'] = FLAGS.num_it
    opts['batch_size'] = FLAGS.batch_size
    opts['lr'] = FLAGS.lr
    opts['plot_every'] = 5000 #int(opts['print_every'] / 2.) + 1
    opts['evaluate_every'] = int(opts['plot_every'] / 10.)
    opts['save_every'] = 10000000000
    opts['save_final'] = FLAGS.save_model
    opts['save_train_data'] = FLAGS.save_data

    # Reset tf graph
    tf.reset_default_graph

    # Loading the dataset
    data = DataHandler(opts)
    assert data.train_size >= opts['batch_size'], 'Training set too small'

    # inti method
    run = Run(opts, data)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        # Dumping all the configs to the text file
        with utils.o_gfile((opts['exp_dir'], 'params.txt'), 'w') as text:
            text.write('Parameters:\n')
            for key in opts:
                text.write('%s : %s\n' % (key, opts[key]))
        run.train()
    else:
        assert False, 'Unknown mode %s' % FLAGS.mode

main()
