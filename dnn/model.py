# -*- coding: utf-8 -*-
import sys
import tensorflow as tf

sys.path.append('...')
from conf import dnn


class modeler(object):
    def __init__(self):
        self.x = dnn.CONFIG['model']['parameter']['input']
        self.model=self.get_model(self.x)
    def get_model(self, x):
        w1 = tf.Variable(dnn.CONFIG['model']['parameter']['w1'], dtype=float, name='w1')
        b1 = tf.Variable(dnn.CONFIG['model']['parameter']['b1'], dtype=float, name='b1')
        w2 = tf.Variable(dnn.CONFIG['model']['parameter']['w2'], dtype=float, name='w2')
        b2 = tf.Variable(dnn.CONFIG['model']['parameter']['b2'], dtype=float, name='b2')
        a = tf.nn.relu(tf.matmul(x, w1) + b1)
        y = tf.sigmoid(tf.matmul(a, w2) + b2)
        result = tf.nn.softmax(y, name='result')
        return result