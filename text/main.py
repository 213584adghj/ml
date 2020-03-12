# -*- coding: utf-8 -*-
import train as tr
import predict as pre

if __name__ == "__main__":
    train = tr.train(5000)  # 训练模型
    predict = pre.predict()  # 预测结果类
    test = train.test_x
    result = predict.work_predict(test)  # 产生结果
