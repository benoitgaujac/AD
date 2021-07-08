"""This class helps to handle the data.

"""

import logging
import tensorflow.compat.v1 as tf
import numpy as np
from math import pi, cos, sin

import configs

import pdb


# def _data_generator(dataset_config):
def _data_generator(dataset_size, dataset, anomalous):
    # dataset_size = dataset_config[0]
    # dataset = dataset_config[1]
    # generate dataset
    n = 0
    while n<dataset_size:
        if anomalous:
            y = np.random.randint(low=0, high=2, size=None, dtype='int32')
        else:
            y = 1
        # generate nominal
        if y==1:
            x = _nominal_generator(dataset)
        else:
            # generate anomalous
            high = 1
            r = np.random.uniform(low=-high, high=high, size=None)
            theta = np.random.uniform(low=0.0, high=pi, size=None)
            x = np.array((r*cos(theta), r*sin(theta)))
        yield x, y
        n+=1

def _nominal_generator(dataset):
    # generate nominal data
    if not isinstance(dataset, str):
        dataset = dataset.decode('UTF-8')
    if dataset=='line':
        r = np.random.uniform(low=-configs.config_line['coef'][0], high=configs.config_line['coef'][0], size=None)
        # r = configs.config_line['coef'][0]
        theta = configs.config_line['theta'] #float(pi / 6.)
        x = np.array((r*cos(theta), r*sin(theta)))
    elif dataset=='quadratic':
        if configs.config_quadratic['fixed_dataset']:
            z = np.random.uniform(low=-0.75, high=0.75, size=None)
        else:
            z = np.random.uniform(low=-10, high=10, size=None)
        # x = np.array((z, configs.config_quadratic['a']*z*z + configs.config_quadratic['b']))
        quad = configs.config_quadratic['coef'][0]*z*z \
                + configs.config_quadratic['coef'][1]*z \
                + configs.config_quadratic['coef'][2]
        x = np.array((z, quad))
        if not configs.config_quadratic['fixed_dataset']:
            x *= np.array([10e-2, 10e-4])
        rot = np.stack([cos(configs.config_quadratic['theta']), -sin(configs.config_quadratic['theta']),
                        sin(configs.config_quadratic['theta']), cos(configs.config_quadratic['theta'])], 0).reshape([2,2])
        x = np.matmul(rot, x)
    elif dataset=='cubic':
        if configs.config_quadratic['fixed_dataset']:
            z = np.random.uniform(low=-0.75, high=0.75, size=None)
        else:
            z = np.random.uniform(low=-10, high=10, size=None)
        # x = np.array((z, configs.config_quadratic['a']*z*z + configs.config_quadratic['b']))
        cub = configs.config_cubic['coef'][0]*z*z*z \
                + configs.config_cubic['coef'][1]*z*z \
                + configs.config_cubic['coef'][2]*z \
                + configs.config_cubic['coef'][3]
        x = np.array((z, cub))
        if not configs.config_quadratic['fixed_dataset']:
            x *= np.array([10e-2, 10e-5])
        rot = np.stack([cos(configs.config_cubic['theta']), -sin(configs.config_cubic['theta']),
                        sin(configs.config_cubic['theta']), cos(configs.config_cubic['theta'])], 0).reshape([2,2])
        x = np.matmul(rot, x)
    else:
        raise ValueError('Unknown {} dataset' % dataset)
    return x


class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        self.dataset = opts['dataset']
        # load data
        logging.error('\n Initialize {}.'.format(self.dataset))
        self._init_dataset(opts)
        logging.error('Initialization Done.')

    def _init_dataset(self, opts):
        """Load a 2d dataset and fill all the necessary variables.

        """
        # dataset size
        self.train_size = 10000
        self.test_size = 10000
        # datashape
        self.data_shape = [2,]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_generator(_data_generator,
                                output_types=(tf.float32, tf.int32),
                                output_shapes=([2,], []),
                                args=(self.train_size, self.dataset, opts['use_anomalous']))
                                # output_signature=(
                                #     tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                #     tf.TensorSpec(shape=(), dtype=tf.int32)),
                                # args=(self.train_size, self.dataset))
        dataset_test = tf.data.Dataset.from_generator(_data_generator,
                                output_types=(tf.float32, tf.int32),
                                output_shapes=([2,], []),
                                args=(self.test_size, self.dataset, opts['use_anomalous']))
                                # output_signature=(
                                #     tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                #     tf.TensorSpec(shape=(), dtype=tf.int32)),
                                # args=(self.test_size, self.dataset))
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = tf.data.make_initializable_iterator(dataset_train)
        self.iterator_test = tf.data.make_initializable_iterator(dataset_test)

        # Global iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        self.next_element = tf.data.Iterator.from_string_handle(
                                self.handle,
                                tf.data.get_output_types(dataset_train),
                                tf.data.get_output_shapes(dataset_train)).get_next()

    def init_iterator(self, sess):
        sess.run([self.iterator_train.initializer,self.iterator_test.initializer])
        # handle = sess.run(iterator.string_handle())
        train_handle, test_handle = sess.run([self.iterator_train.string_handle(),self.iterator_test.string_handle()])
        return train_handle, test_handle

    def _sample_observation(self, batch_size, dataset, nominal):
        if nominal:
            obs = np.zeros([batch_size,2])
            for n in range(batch_size):
                # x = _nominal_generator(dataset)
                obs[n] = _nominal_generator(dataset)
            labels = np.ones([batch_size,])
        else:
            high = 1e2
            r = np.random.uniform(low=-high, high=high, size=batch_size)
            theta = np.random.uniform(low=0.0, high=pi, size=batch_size)
            obs = np.stack((r*np.cos(theta), r*np.sin(theta)),axis=-1)
            labels = np.zeros([batch_size,])
        return obs, labels
