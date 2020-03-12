# -*- coding: utf-8 -*-
import sys

sys.path.append('...')
from conf import easy_linear_regression as elr
import data as da


class train(object):
    def __init__(self):
        lr = 0.0001
        epoch_number = elr.CONFIG['train']['epoch_numbe']
        w = 0.
        b = 0.
        data = da.data()
        self.data = da.data
        N = len(data.x)
        points = []
        points.append(data.x)
        points.append(data.y)
        for i in range(epoch_number):
            current_w, current_b = 0., 0.
            for i in range(0, N):
                current_w = current_w + 2 / N * (w * points[i, 0] - points[i, 1]) * points[i, 0]
                current_b = current_b + 2 / N * (w * points[i, 0] - points[i, 1])
            w = w - lr * current_w

            b = b - lr * current_b
