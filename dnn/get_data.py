# -*- coding: utf-8 -*-
import sys
from numpy import *
import numpy as np

sys.path.append('...')
from conf import dnn


class data(object):
    def __init__(self):
        self.train_data_amount = dnn.CONFIG['data']['all_data_amount']
        self.train_future, self.train_label = self.birth_data(dnn.CONFIG['data']['train_data_amount'])
        self.test_future, self.test_label = self.birth_data(dnn.CONFIG['data']['test_data_amount'])

    def birth_data(self, T):
        data = []
        label = []
        for i in range(T):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            if x == 0 or y == 0:
                continue
            if x > 0 and y > 0:
                label.append(np.array([1, 0, 0, 0]))
            if x > 0 and y < 0:
                label.append(np.array([0, 0, 0, 1]))
            if x < 0 and y > 0:
                label.append(np.array([0, 1, 0, 0]))
            if x < 0 and y < 0:
                label.append(np.array([0, 0, 1, 0]))
            v = []
            v.append(x)
            v.append(y)
            data.append(v)
        label = np.array(label)
        return np.array(data), np.array(label)
    def save_data(self):
        return