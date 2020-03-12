# -*- coding: utf-8 -*-
import sys
import math

sys.path.append('...')
from conf import lstm
import csv
import numpy as np


# 获得原始数据，生成训练数据，测试数据
class data(object):
    def __init__(self):
        self.base_path = lstm.CONFIG['data']['base_path']  # 所有数据的存储位置
        self.test_path = lstm.CONFIG['data']['test_path']  # 处理原始数据之后得到的测试数据目录
        self.train_path = lstm.CONFIG['data']['train_path']  # 处理原始数据之后得到的训练数据目录
        # self.make_test_train_file()
        self.cbwd_dict = self.get_cbwd_dict()
        self.train_y = self.get_train_y()
        self.train_x = self.get_train_x()

    def make_test_train_file(self):
        result_test = []
        result_train = []
        with open(self.base_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            for i in rows:
                if (i[5] == 'NA'):
                    i.remove(i[5])
                    result_test.append(i)
                else:
                    result_train.append(i)
        test_out = open(self.test_path, 'a', newline='', encoding='utf-8')
        csv_write = csv.writer(test_out, dialect='excel')
        for i in result_test:
            csv_write.writerow(i)
        train_out = open(self.train_path, 'a', newline='', encoding='utf-8')
        csv_write = csv.writer(train_out, dialect='excel')
        for i in result_train:
            csv_write.writerow(i)

    def get_train_y(self):
        result = []
        with open(self.train_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            for i in range(1, len(rows)):
                result.append(float(rows[i][5]))
        return np.array(result)

    def get_cbwd_dict(self):  # 原始数据中cbwd这一列对应着非数值数据，所以需要类型转换
        result = {}
        with open(self.base_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            count = 1
            for i in rows:
                if i[9] not in result:
                    result[i[9]] = count
                    count += 1
        return result

    def get_train_x(self):
        result = []
        with open(self.train_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            for i in rows:
                if (i[1] == 'year'):
                    continue
                k = i
                k[9] = self.cbwd_dict[k[9]]
                result.append(k[4:-1])
        return np.array(result, dtype=float)
