# -*- coding: utf-8 -*-
import data as da
import tensorflow as tf
import sys

sys.path.append('...')
from conf import easy


class train(object):
    def __init__(self):
        self.minist = da.data().minist
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.prediction, self.loss, self.train_step, self.accuracy = self.get_model()
        self.train_model()
        pass

    def get_model(self):
        # 创建一个简单的神经网络（无隐藏层）
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        prediction = tf.nn.softmax(tf.matmul(self.x, W) + b)

        # 二次代价函数
        # loss = tf.reduce_mean(tf.square(y-prediction))
        # 交叉熵
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction))

        # 使用梯度下降法
        learn_rate = easy.CONFIG['train']['learn_rate']
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

        # 初始化变量
        # init = tf.global_variables_initializer()

        #
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(prediction, 1))
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return prediction, loss, train_step, accuracy  # 预测，损失函数，训练，准确率

    def train_model(self):
        batch_size = 100
        # 计算一共有多少个批次
        n_batch = self.minist.train.num_examples // batch_size
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(21):
                for batch in range(n_batch):
                    batch_xs, batch_ys = self.minist.train.next_batch(batch_size)
                    sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y: batch_ys})
                acc = sess.run(self.accuracy,
                               feed_dict={self.x: self.minist.test.images, self.y: self.minist.test.labels})
                print("Iter" + str(epoch) + ",Testing Accuracy" + str(acc))

        pass



