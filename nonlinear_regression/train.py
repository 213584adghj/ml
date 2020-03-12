# coding: utf-8
import sys

sys.path.append('...')
import tensorflow as tf
from conf import nonlinear_regression as nr
import data as da
import matplotlib.pyplot as plt


class train(object):
    def __init__(self):
        self.data = da.data()
        self.x = tf.placeholder(tf.float32, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.get_model_train()
        pass

    def get_model_train(self):
        # 定义神经网络中间层
        Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
        biases_L1 = tf.Variable(tf.zeros([1, 10]))
        Wx_plus_b_L1 = tf.matmul(self.x, Weights_L1) + biases_L1
        L1 = tf.nn.tanh(Wx_plus_b_L1)

        # 定义神经网络输出层
        Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
        biases_L2 = tf.Variable(tf.zeros([1, 1]))
        Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
        prediction = tf.nn.tanh(Wx_plus_b_L2)

        # 二次代价函数
        loss = tf.reduce_mean(tf.square(self.y - prediction))
        # 使用梯度下降训练法
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        with tf.Session() as sess:
            # 变量初始化
            sess.run(tf.global_variables_initializer())
            for _ in range(2000):
                sess.run(train_step, feed_dict={self.x: self.data.x_data, self.y: self.data.y_data})

            # 获得预测值
            prediction_value = sess.run(prediction, feed_dict={self.x: self.data.x_data})
            # 画图
            plt.figure()
            plt.scatter(self.data.x_data, self.data.y_data)
            plt.plot(self.data.x_data, prediction_value, 'r-', lw=5)
            plt.show()


a = train()
