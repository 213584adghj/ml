# -*- coding: utf-8 -*-
import sys

sys.path.append('...')
from sklearn.externals import joblib
from conf import text


class predict(object):
    def __init__(self):
        self.path = text.CONFIG['train']['model']['classification_model_save_path']
        self.model = joblib.load(self.path)
        pass

    def work_predict(self, x):
        return self.model.predict(x)

    def roc(self, predict_y, real_y):
        pass
