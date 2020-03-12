# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import sys

sys.path.append('...')
from conf import cnn_minist as easy


class data(object):
    def __init__(self):
        self.data_name = easy.CONFIG['data']['name']
        self.minist = input_data.read_data_sets(self.data_name, one_hot=True)
