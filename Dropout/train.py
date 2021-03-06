# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# import numpy as np
class train(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.prediction, self.loss, self.train_step, self.accuracy=self.get_model()
        self.train_model()

    def get_model(self):
        W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
        b1 = tf.Variable(tf.zeros([2000]) + 0.1)
        L1 = tf.nn.tanh(tf.matmul(self.x, W1) + b1)
        L1_drop = tf.nn.dropout(L1, self.keep_prob)

        W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
        b2 = tf.Variable(tf.zeros([2000]) + 0.1)
        L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
        L2_drop = tf.nn.dropout(L2, self.keep_prob)

        W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
        b3 = tf.Variable(tf.zeros([1000]) + 0.1)
        L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
        L3_drop = tf.nn.dropout(L3, self.keep_prob)

        W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
        b4 = tf.Variable(tf.zeros([10]) + 0.1)
        prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

        # 二次代价函数
        # loss = tf.reduce_mean(tf.square(y-prediction))
        # 交叉熵
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction))

        # 使用梯度下降法
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

        # 初始化变量
        # init = tf.global_variables_initializer()

        #
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(prediction, 1))
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return prediction, loss, train_step, accuracy

    def train_model(self):
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        # 每个批次的大小
        batch_size = 100
        # 计算一共有多少个批次
        n_batch = mnist.train.num_examples // batch_size
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(31):
                for batch in range(n_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 0.7})

                test_acc = sess.run(self.accuracy, feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels, self.keep_prob: 1.0})
                train_acc = sess.run(self.accuracy, feed_dict={self.x: mnist.train.images, self.y: mnist.train.labels, self.keep_prob: 1.0})
                print("Iter" + str(epoch) + ",Testing Accuracy" + str(test_acc) + ",Training Accuracy" + str(train_acc))

        pass
a=train()