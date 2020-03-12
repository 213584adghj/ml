# -*- coding:utf-8 -*-
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import get_data as gt

sys.path.append('...')
from conf import knn as kn
from sklearn.externals import joblib


class trainer():
    def __init__(self):
        self.data = gt.data()
        self.parameter = kn.CONFIG['train']['parameter']
        self.model=self.get_model()
    def get_model(self):
        clf = KNeighborsClassifier(n_neighbors=self.parameter['n_neighbors'],
                                   algorithm=self.parameter['algorithm'])
        clf.fit(self.data.x_train, self.data.y_train)
        #joblib.dump(self.model, kn.CONFIG['train']['result_model_path'])
        return clf