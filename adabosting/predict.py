# -*- coding:utf-8 -*-
import os
import sys
from sklearn.externals import joblib
from collections import Counter
import matplotlib.pyplot as plt
import getdata  as gt
sys.path.append('...')
from conf import ada_bosting as ada
class predicter(object):
    def __init__(self):
        data=gt.data()
        self.X=data.test_x
        self.Y=data.test_y
        #print(self.Y)
        q=joblib.load(self.get_path())
        #print(q.predict(self.X))
        self.Roc(q.predict(self.X),self.Y)
    def get_path(self):
            q = (os.path.abspath('..'))
            q = (q.rstrip('src/adabosting'))
            path = q + ada.CONFIG['train']['target_path']
            return path

    def Roc(self,label_test, label_real):
        result_x = []
        result_y = []
        result_x.append(0)
        result_y.append(0)
        res = Counter(label_real)
        mz = res[1]
        mf = res[0]
        x = 0.
        y = 0.
        for i in range(len(label_real) - 1):
            if (label_real[i] ==  1 and label_test[i] == 1):  # 当前样例为真正例
                y = y + 1. / float(mz)
                result_x.append(x)
                result_y.append(y)
            if (label_test[i] == 1 and label_real[i] == 0):  # 当前样例为假正例
                x = x + 1. / float(mf)
                result_y.append(y)
                result_x.append(x)

        fig = plt.figure()
        plt.plot(result_x, result_y)
        plt.show()
