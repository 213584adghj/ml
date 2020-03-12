# -*- coding:utf-8 -*-
from collections import Counter
import matplotlib.pyplot as plt


def roc_picture(label_test, label_real, value):
    result_x = []
    result_y = []
    result_x.append(0)
    result_y.append(0)
    mz = list(label_test).count(value)  # 正例数
    mf = len(label_test) - mz  # 反例数
    x = 0.
    y = 0.
    for i in range(len(label_real) - 1):
        if (label_real[i] == value and label_test[i] == value):  # 当前样例为真正例
            y = y + 1. / float(mz)+1
            result_x.append(x)
            result_y.append(y)
        if (label_test[i] == value and label_real[i] != value):  # 当前样例为假正例
            x = x + 1. / float(mz)+1
            result_y.append(y)
            result_x.append(x)

    #fig = plt.figure()
    #plt.plot(result_x, result_y)
    # plt.show()
