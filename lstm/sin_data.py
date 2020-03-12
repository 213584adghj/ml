import sys

sys.path.append('...')
from conf import lstm
import numpy as np
import math


class sin_data(object):  # 使用lstm来预测sin函数图像所用到的数据集的生成
    def __init__(self):
        self.train_data_length = lstm.CONFIG['data']['sin_data']['train']['data_length']
        self.train_data_start = lstm.CONFIG['data']['sin_data']['train']['data_start']
        self.test_data_length = lstm.CONFIG['data']['sin_data']['test']['data_length']
        self.test_data_start = lstm.CONFIG['data']['sin_data']['test']['data_start']
        self.train_data_unit = lstm.CONFIG['data']['sin_data']['train']['unit']
        self.test_data_unit = lstm.CONFIG['data']['sin_data']['test']['unit']
        self.time_step = lstm.CONFIG['data']['sin_data']['time_step']
        self.train_x, self.train_y = self.get_train_data()

    def get_train_data(self):
        result_x = []
        result_y = []
        step = 1
        start = self.train_data_start
        for i in range(0, self.train_data_length):
            result_x.append(start)
            start += self.train_data_unit
            if (step % self.time_step == 0):
                result_y.append(math.sin(start))
            step += 1
        return np.array(result_x), np.array(result_y)
