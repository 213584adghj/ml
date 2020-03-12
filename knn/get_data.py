# -*- coding: utf-8 -*-
import numpy as np
import sys
from sklearn.model_selection import train_test_split

sys.path.append('...')
from conf import knn as kn
import os
import re


class data(object):
    def __init__(self):
        self.base_data_path = self.get_path()
        self.all_data_x, self.all_data_y = self.get_all_data()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_all_data()
        self.save_train_path()
        self.save_test_path()

    def get_path(self):
        path = os.getcwd()
        path = path.rstrip('\src\knn')
        path = path + kn.CONFIG['data']['base_data_file']
        return path

    def get_all_data(self):
        dataMat = []
        labelMat = []
        path = self.base_data_path
        fr = open(path, 'r')
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            dataMat.append(np.array(curLine[0:len(curLine) - 1], dtype=float))
            labelMat.append(int(re.sub("\D", "", curLine[-1])))
        return np.array(dataMat), np.array(labelMat, dtype=int)

    def split_all_data(self):
        test_size = 1 - kn.CONFIG['data']['parameter']['characteristic_amount']
        x_train, x_test, y_train, y_test = train_test_split(self.all_data_x, self.all_data_y, test_size=test_size,
                                                            random_state=0)
        return x_train, x_test, y_train, y_test

    def save_train_path(self):
        path = os.getcwd()
        path = path.rstrip('\\src\\knn')
        p = path + kn.CONFIG['data']['train_data_path']['x']
        q = path + kn.CONFIG['data']['train_data_path']['y']
        np.savetxt(p, self.x_train, fmt='%s', newline='\n')
        np.savetxt(q, self.y_train, fmt='%s', newline='\n')

    def save_test_path(self):
        path = os.getcwd()
        path = path.rstrip('\\src\\knn')
        p = path + kn.CONFIG['data']['test_data_path']['x']
        q = path + kn.CONFIG['data']['test_data_path']['y']
        np.savetxt(p, self.x_test, fmt='%s', newline='\n')
        np.savetxt(q, self.y_test, fmt='%s', newline='\n')