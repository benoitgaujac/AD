import os
from datetime import datetime
import logging
import argparse
import configs
from train import Run
from datahandler import DataHandler
import utils

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import pdb

parser = argparse.ArgumentParser()
# run setup
parser.add_argument("--mode", default='train',
                    help='mode to run [train/vizu/fid/test]')
parser.add_argument("--dataset", default='line',
                    help='dataset')
parser.add_argument("--num_it", type=int, default=10000,
                    help='iteration number')
parser.add_argument("--batch_size", type=int, default=100,
                    help='batch size')
parser.add_argument("--lr", type=float, default=0.01,
                    help='learning rate size')
# exp setup
parser.add_argument("--exp_id", type=int,
                    help='exp. id')
# path setup
parser.add_argument("--res_dir", type=str,
                    help='root directory for res')
parser.add_argument("--out_dir", type=str, default='code_outputs',
                    help='dir in which experiment outputs are saved')
# model set up
parser.add_argument("--model", default='affine',
                    help='score model to train [affine/nonaffine]')
parser.add_argument("--scr_nonlin", default='linear',
                    help='non linear activation for score fct')
parser.add_argument("--train_w", action='store_false', default=True,
                    help='whether to learn linear proj')
parser.add_argument("--beta", type=float, default=0.,
                    help='weight regulation')
parser.add_argument("--train_d", action='store_false', default=True,
                    help='whether to learn D')
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
    elif FLAGS.dataset == 'quadratic':
        opts = configs.config_quadratic
    else:
        raise ValueError('Unknown {} dataset' % FLAGS.dataset)

    # Model set up
    opts['model'] = FLAGS.model
    opts['score_non_linear'] = FLAGS.scr_nonlin
    opts['train_w'] = FLAGS.train_w
    if opts['train_w']:
        opts['beta'] = FLAGS.beta
    else:
        opts['beta'] = 0.
    opts['train_d'] = FLAGS.train_d
    if opts['train_d']:
        lambdas = [0., 0.1, 1.]
        exp_id = (FLAGS.exp_id-1) % len(lambdas)
        opts['lmbda'] = lambdas[exp_id]
    else:
        opts['lmbda'] = 0.

    # Create directories
    if FLAGS.res_dir:
            results_dir = FLAGS.res_dir
    else:
        results_dir = 'results'
    if not tf.io.gfile.isdir(results_dir):
        utils.create_dir(results_dir)
    out_dir = os.path.join(results_dir, opts['model'])
    if not tf.io.gfile.isdir(out_dir):
        utils.create_dir(out_dir)
    opts['out_dir'] = os.path.join(out_dir, FLAGS.out_dir)
    if not tf.io.gfile.isdir(opts['out_dir']):
        utils.create_dir(opts['out_dir'])
    if FLAGS.res_dir:
        exp_name = FLAGS.res_dir + '_'
    else:
        exp_name = ''
    exp_name += 'beta' + str(opts['beta']) + '_lambda' + str(opts['lmbda'])
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
    opts['plot_every'] = 10000 #int(opts['print_every'] / 2.) + 1
    opts['evaluate_every'] = int(opts['plot_every'] / 5.) #int(opts['print_every'] / 2.) + 1
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
