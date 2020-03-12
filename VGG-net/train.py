# coding: utf-8
import _pickle as cPickle
import os
import numpy as np
import data as da
import tensorflow as tf


class train(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 3072])
        self.y = tf.placeholder(tf.int64, [None])
        self.loss, self.predict, self.accuracy, self.train_op = self.get_model()
        self.train_model()
        pass

    def get_model(self):
        # 卷积部分
        x_image = tf.reshape(self.x, [-1, 3, 32, 32])
        x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])
        conv1_1 = tf.layers.conv2d(x_image,
                                   32,  # output channel number
                                   (3, 3),  # kernel size
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv1_1')
        conv1_2 = tf.layers.conv2d(conv1_1,
                                   32,  # output channel number
                                   (3, 3),  # kernel size
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv1_2')
        pooling1 = tf.layers.max_pooling2d(conv1_2,
                                           (2, 2),  # kernel size
                                           (2, 2),  # size
                                           name='pool1')
        conv2_1 = tf.layers.conv2d(pooling1,
                                   32,  # output channel number
                                   (3, 3),  # kernel size
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv2_1')
        conv2_2 = tf.layers.conv2d(conv2_1,
                                   32,  # output channel number
                                   (3, 3),  # kernel size
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv2_2')
        pooling2 = tf.layers.max_pooling2d(conv2_2,
                                           (2, 2),  # kernel size
                                           (2, 2),  # size
                                           name='pool2')
        conv3_1 = tf.layers.conv2d(pooling2,
                                   32,  # output channel number
                                   (3, 3),  # kernel size
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv3_1')
        conv3_2 = tf.layers.conv2d(conv3_1,
                                   32,  # output channel number
                                   (3, 3),  # kernel size
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv3_2')
        pooling3 = tf.layers.max_pooling2d(conv3_2,
                                           (2, 2),  # kernel size
                                           (2, 2),  # size
                                           name='pool3')
        # [None, 4*4*32]
        flatten = tf.layers.flatten(pooling3)
        y_ = tf.layers.dense(flatten, 10)

        """
        # [None]
        y = tf.placeholder(tf.int64,[None])

        hidden1 = tf.layers.dense(x,100,activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1,100,activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2,50,activation=tf.nn.relu)
        y_ = tf.layers.dense(hidden3,10)
        """

        # 交叉熵损失函数
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=y_)

        predict = tf.argmax(y_, 1)
        # bool型
        correct_prediction = tf.equal(predict, self.y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

        with tf.name_scope('train_op'):
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

        return loss, predict, accuracy, train_op

    def train_model(self):
        init = tf.global_variables_initializer()
        batch_size = 20
        train_steps = 1000
        test_steps = 100
        data = da.data()
        train_data = data.train_data
        test_data = data.train_data
        with tf.Session() as sess:
            sess.run(init)
            for i in range(train_steps):
                batch_data, batch_labels = train_data.next_batch(batch_size)
                loss_val, acc_val, _ = sess.run([self.loss, self.accuracy, self.train_op],
                                                feed_dict={self.x: batch_data, self.y: batch_labels})
                if (i + 1) % 500 == 0:
                    print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i + 1, loss_val, acc_val))
                if (i + 1) % 5000 == 0:
                    all_test_acc_val = []
                    for j in range(test_steps):
                        test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                        test_acc_val = sess.run([self.accuracy],
                                                feed_dict={self.x: test_batch_data, self.y: test_batch_labels})
                        all_test_acc_val.append(test_acc_val)
                    test_acc = np.mean(all_test_acc_val)
                    print('[Test] Step: %d,acc: %4.5f' % (i + 1, test_acc))
        pass
# a=train()
