# -*- coding: utf-8 -*-
import data as da
import tensorflow as tf


class train(object):
    def __init__(self):
        self.mnist = da.data().minist
        # 定义两个placeholder
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.prediction, self.cross_entropy, self.train_step, self.accuracy, self.keep_prob = self.get_model()
        self.train_model()
        pass

    def get_model(self):
        # 每个批次的大小
        batch_size = 100
        # 计算一共多少个批次
        n_batch = self.mnist.train.num_examples // batch_size

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
            return tf.Variable(initial)

        # 初始化偏置
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # 卷积层
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        # 池化层
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 初始化权值
        # 改变x的格式转化为4D的向量
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        # 初始化第一个卷积层的权值和偏置
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        # 把x_image和权值向量进行卷积，再加上偏置值，然后应用relu激活函数
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)  # 进行max_pooling

        # 初始化第二个卷积层的权值和偏置
        W_conv2 = weight_variable([5, 5, 32, 64])  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
        b_conv2 = bias_variable([64])  # 每一个卷积核一个偏置值

        # h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

        # 初始化第一个全连接层的权值
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        # 把池化层2的输出扁平化为1维
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        # 求第一个全连接层的输出
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # keep_prob用来表示神经元的输出概率
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 初始化第二个全连接层
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        # 计算输出
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction))
        # 使用AdamOptimizer进行优化
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return prediction, cross_entropy, train_step, accuracy, keep_prob  # 预测，损失函数，训练，准确率

    def train_model(self):
        batch_size = 100
        n_batch = self.mnist.train.num_examples // batch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(21):
                for batch in range(n_batch):
                    batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                    sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 0.7})

                acc = sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images, self.y: self.mnist.test.labels,
                                                         self.keep_prob: 1.0})
                print("Iter" + str(epoch) + ",Testing Accuracy=" + str(acc))

        pass
