import tensorflow.compat.v1 as tf
import os
import sys
import copy
import numpy as np
import logging

import pdb


class File(tf.io.gfile.GFile):
    """Wrapper on GFile extending seek, to support what python file supports."""
    def __init__(self, *args):
        super(File, self).__init__(*args)

    def seek(self, position, whence=0):
        if whence == 1:
            position += self.tell()
        elif whence == 2:
            position += self.size()
        else:
            assert whence == 0
        super(File, self).seek(position)

def o_gfile(filename, mode):
    """Wrapper around file open, using gfile underneath.

    filename can be a string or a tuple/list, in which case the components are
    joined to form a full path.
    """
    if isinstance(filename, tuple) or isinstance(filename, list):
        filename = os.path.join(*filename)
    return File(filename, mode)

def create_dir(d):
    if not tf.io.gfile.isdir(d):
        tf.io.gfile.mkdir(d)

def listdir(dirname):
    return tf.io.gfile.ListDirectory(dirname)
