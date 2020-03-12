# -*- coding:utf-8 -*-
import sys

sys.path.append('...')
import sys


class data(object):
    def __init__(self):
        self.data_path = self.get_path()
        self.all_data = self.get_data(self.data_path)

    def get_path(self):
        path = sys.path[1]
        a = (path + '\\data\\k_means_data\\testSet.txt')
        return a

    def get_data(self, path):
        fr = open(path, 'r')
        data = []
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            # print(curLine)
            one_result = []
            for i in curLine:
                one_result.append(float(i))
            data.append(one_result)
        return data
