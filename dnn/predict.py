# -*- coding: utf-8 -*-
import getdata as gt
import numpy as np
import model
import sys
import tensorflow as tf

from conf import dnn

sys.path.append('...')


class predict_(object):  # 预测类
    def __init__(self, data):
        self.predict()
        self.data = data

    def predict_(self):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('abc.meta')  # 加载图结构
            gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
            tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
