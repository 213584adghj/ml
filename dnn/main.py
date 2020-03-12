# -*- coding: utf-8 -*-
import model
import get_data as gt
import train
import sys
import presict as pre

sys.path.append('...')
from conf import dnn

if __name__ == "__main__":
    q = train()  # 训练并且将模型保存
    u = gt.data()
    test_data = u.birth_data(dnn.CONFIG['predict']['amount'])
    result = pre.predict()
    pre.predict_()  # 加载模型并且预测预测
