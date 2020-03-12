# -*- coding:utf-8 -*-
import sys
import os

sys.path.append('...')
from conf import ada_bosting as ada

from sklearn.ensemble import AdaBoostClassifier
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
        algorithm = ada.CONFIG['train']['parameter']['algorithm']
        n_estimators = ada.CONFIG['train']['parameter']['n_estimators']
        learning_rate = ada.CONFIG['train']['parameter']['learning_rate']
        model = AdaBoostClassifier(algorithm=algorithm, n_estimators=n_estimators,
                                   learning_rate=learning_rate)
        model.fit(x, y)
        return model
