# -*- coding:utf-8 -*-
import sys
import os

sys.path.append('...')
from conf import ada_bosting as ada

from sklearn.linear_model import LogisticRegression
import getdata as  gt
from sklearn.externals import joblib


class trainer(object):
    def __init__(self):
        self.data = gt.data()
        self.path = self.get_path()
        self.model = self.train(self.data.train_x, self.data.train_y)
        joblib.dump(self.model, self.path)

    def get_path(self):
        q = (os.path.abspath('..'))
        q = (q.rstrip('src/adabosting'))
        path = q + ada.CONFIG['train']['target_path']
        return path

    def train(self, x, y):
        model = LogisticRegression()
        model.fit(x, y)
        return model
