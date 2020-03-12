# coding: utf-8
import sys
import numpy as np

sys.path.append('...')
from conf import nonlinear_regression as nr


class data(object):
    def __init__(self):
        self.x_start = nr.CONFIG['data']['x_start']
        # print(self.x_start)
        self.x_end = nr.CONFIG['data']['x_end']
        # print(self.x_end)
        self.x_length = nr.CONFIG['data']['x_length']
        # print(self.x_length)
        self.noise_start = nr.CONFIG['data']['noise_start']
        # print(self.noise_start)
        self.noise_end = nr.CONFIG['data']['noise_end']
        # print(self.noise_end)
        self.x_data = np.linspace(self.x_start, self.x_end, self.x_length)[:, np.newaxis]
        # print(self.x_start)
        self.noise = np.random.normal(self.noise_start, self.noise_end, self.x_data.shape)
        # print(self.x_start)
        self.y_data = np.square(self.x_data) + self.noise


a = data()
