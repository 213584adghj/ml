# -*- coding:utf-8 -*-
import numpy as np
import re
import sys
import os

sys.path.append('...')
from conf import ada_bosting as ada


class data(object):
    def __init__(self):
        q = (os.path.abspath('..'))
        q = (q.rstrip('src/adabosting'))
        self.train_path = q + 'data/adabosting_data' + '/' + ada.CONFIG['data']['train_path']
        self.test_path = q + 'data/adabosting_data' + '/' + ada.CONFIG['data']['train_path']
        self.train_x, self.train_y = self.get(self.train_path)
        self.test_x, self.test_y = self.get(self.test_path)

    def get(self, name):
        dataMat = []
        labelMat = []
        path = name
        fr = open(path, 'r')
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            dataMat.append(np.array(curLine[0:len(curLine) - 1], dtype=float))
            labelMat.append(int(re.sub("\D", "", curLine[-1])))
        return np.array(dataMat), np.array(labelMat, dtype=int)
