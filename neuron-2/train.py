# coding: utf-8
import _pickle as cPickle
import os
import numpy as np
import data as da
import tensorflow as tf


class train(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 3072])
        # [None]
        self.y = tf.placeholder(tf.int64, [None])
        self.loss, self.predict, self.accuracy, self.train_op = self.get_model()
        self.train_model()
        # (3072,1)
        pass

    def get_model(self):
        w = tf.get_variable('w', [self.x.get_shape()[-1], 1],
                            initializer=tf.random_normal_initializer(0, 1))
        # (1,)
        b = tf.get_variable('b', [1],
                            initializer=tf.constant_initializer(0.0))
        # [None,3072]*[3072,1] = [None,1]
        y_ = tf.matmul(self.x, w) + b
        # [None,1]
        p_y_1 = tf.nn.sigmoid(y_)
        # [None,1]
        y_reshaped = tf.reshape(self.y, (-1, 1))
        y_reshaped_float = tf.cast(y_reshaped, tf.float32)

        loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

        predict = p_y_1 > 0.5
        # bool型
        correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        with tf.name_scope('train_op'):
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return loss, predict, accuracy, train_op

    def train_model(self):
        init = tf.global_variables_initializer()
        batch_size = 20
        train_steps = 100000
        test_steps = 100
        data = da.data()
        train_data = data.train_data
        test_data = data.train_data
        with tf.Session() as sess:
            sess.run(init)
            for i in range(train_steps):
                batch_data, batch_labels = train_data.next_batch(batch_size)
                loss_val, acc_val, _ = sess.run([self.loss, self.accuracy, self.train_op], feed_dict={self.x: batch_data, self.y: batch_labels})
                if (i + 1) % 500 == 0:
                    print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i + 1, loss_val, acc_val))
                if (i + 1) % 5000 == 0:
                    #test_data = CifarData(test_filenames, False)
                    all_test_acc_val = []
                    for j in range(test_steps):
                        test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                        test_acc_val = sess.run([self.accuracy], feed_dict={self.x: test_batch_data, self.y: test_batch_labels})
                        all_test_acc_val.append(test_acc_val)
                    test_acc = np.mean(all_test_acc_val)
                    print('[Test] Step: %d,acc: %4.5f' % (i + 1, test_acc))
# a=train()
