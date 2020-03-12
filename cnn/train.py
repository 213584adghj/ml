# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from numpy import *
import sys
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('...')


class _train():
    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        x = tf.placeholder(tf.float32, [None, 784])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        conv1_weights = tf.get_variable('wes', [5, 5, 1, 6])
        conv1_bases = tf.get_variable('bas', [6])
        conv1_result = tf.nn.conv2d(x_image,
                                    conv1_weights,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
        convv1_finalresult = tf.nn.max_pool(tf.nn.relu(
            tf.nn.bias_add(conv1_result, conv1_bases)),
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="VALID")
        convv2_input = convv1_finalresult
        conv2_weights = tf.get_variable('wexs', [5, 5, 6, 16])
        conv2_bases = tf.get_variable('basq', [16])
        conv2_result = tf.nn.conv2d(convv2_input,
                                    conv2_weights,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
        conv2_finalresult = tf.nn.max_pool(tf.nn.relu(
            tf.nn.bias_add(conv2_result, conv2_bases)),
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="VALID")

        op = conv2_finalresult.get_shape().as_list()

        fc1_input = tf.reshape(conv2_finalresult, [-1, 144])

        fc1_weights = tf.get_variable('xe', [144, 120], dtype=float)
        fc1_bases = tf.get_variable('dx', [120], dtype=float)
        fc1_result = tf.nn.relu(tf.matmul(fc1_input, fc1_weights) + fc1_bases)

        fc2_input = fc1_result
        fc2_weights = tf.get_variable('xfe', [120, 84], dtype=float)
        fc2_bases = tf.get_variable('dxq', [84], dtype=float)
        fc2_result = tf.nn.relu(tf.matmul(fc2_input, fc2_weights) + fc2_bases)

        fc3_input = fc2_result
        fc3_weights = tf.get_variable('xze', [84, 10], dtype=float)
        fc3_result = tf.nn.softmax(tf.sigmoid(tf.matmul(fc3_input, fc3_weights)))
        L = tf.placeholder(tf.float32, [None, 10])
        loss = -tf.reduce_mean(L * tf.log(tf.clip_by_value(fc3_result, 0, 1)))
        train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # print(mnist.train.num_examples)

            for i in range(2000):
                c, d = mnist.train.next_batch(500)
                sess.run(train, feed_dict={x: c, L: d})
                # print(sess.run(loss, feed_dict={x: c, L: d}))

            testa, testb = mnist.train.next_batch(mnist.test.num_examples)

            wq = np.array(sess.run(fc3_result, feed_dict={x: testa}))

            RESULT = []

            for s in wq:
                RESULT.append(np.argmax(s))

            qbz = []
            wsx = np.array(testb)

            for s in wsx:
                qbz.append(np.argmax(s))
            count1 = 0

            for i in range(len(RESULT)):
                if (RESULT[i] == qbz[i]):
                    count1 += 1
