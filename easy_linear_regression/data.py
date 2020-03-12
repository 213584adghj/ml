# -*- coding: utf-8 -*-
import sys
import random

sys.path.append('...')
from conf import easy_linear_regression as elr


class data(object):
    def __init__(self):  # count表示数据的数量
        self.x = []  # 生成x
        self.y = []  # 生成y
        noise = random.randint(-5, 5)  # 生成噪音
        start = elr.CONFIG['data']['start']
        end = elr.CONFIG['data']['end']
        w = elr.CONFIG['data']['w']
        b = elr.CONFIG['data']['b']
        for i in range(start, end):
            self.x.append(i)
            self.y.append(w * i + b + noise)
        return 
