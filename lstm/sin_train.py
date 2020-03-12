# -*- coding:utf-8 -*-
import sys

sys.path.append('...')
from conf import lstm
import numpy as np
import tensorflow as tf


class sin_train(object):
    def __init__(self):
        train_examples = 10000
        test_examples = 1000
        time_step = 10
        sample_gap = 0.01
        test_start = (train_examples + time_step) * sample_gap
        test_end = test_start + (test_examples + time_step) * sample_gap
        train_x, train_y = self.generate_data(
            np.sin(np.linspace(0, test_start, train_examples + time_step, dtype=np.float32)))
        self.test_x, self.test_y = self.generate_data(
            np.sin(np.linspace(test_start, test_end, test_examples + time_step, dtype=np.float32)))

        # 开始训练模型，创建会话
        with tf.Session() as sess:
            self.trian(sess, train_x, train_y)
            a, b, c = self.lstm_model(train_x, train_y)
            print(a)
            self.run_eval(sess, self.test_x, self.test_y)
        return

    def generate_data(self, seq):
        X = []
        y = []
        for i in range(len(seq) - 10):
            X.append([seq[i:i + 10]])
            y.append([seq[i + 10]])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def lstm_model(self, X, y):
        # 使用多层的LSTM结构
        hidden_size = lstm.CONFIG['sin_train']['lstm']['hidden_size']
        num_layers = lstm.CONFIG['sin_train']['lstm']['num_layers']
        forget_bias = lstm.CONFIG['sin_train']['lstm']['forget_bias']
        state_is_tuple = lstm.CONFIG['sin_train']['lstm']['state_is_tuple']
        activation_fn = lstm.CONFIG['sin_train']['lstm']['activation_fn']
        optimizer = lstm.CONFIG['sin_train']['lstm']['optimizer']
        learning_rate = lstm.CONFIG['sin_train']['lstm']['learning_rate']
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=state_is_tuple) for _ in
             range(num_layers)])
        # cell_initializer = cell.zero_state(batch_size,tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        # outputs[batch_size,-1,:]==state[1,batch_size,:]
        output = outputs[:, -1, :]  # state[1]
        # 对LSTM网络的输出再加一层全连接层，不设置激活函数
        prediction = tf.contrib.layers.fully_connected(output, 1, activation_fn=activation_fn)
        # 计算损失函数
        loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
        # 创建优化器
        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer=optimizer,
                                                   learning_rate=learning_rate)
        return prediction, loss, train_op

    def trian(self, sess, train_x, train_y):
        batch_size = 32
        # 将训练数据一数据集的形式提供给计算图
        ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        ds = ds.repeat().shuffle(1000).batch(batch_size)
        X, y = ds.make_one_shot_iterator().get_next()

        # 调用模型，得到预测结果，损失函数以及训练操作
        with tf.variable_scope("model"):
            prediction, loss, train_op = self.lstm_model(X, y)

        # 初始化变量
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            train_, l = sess.run([train_op, loss])
            if i % 100 == 0:
                print("train_step:{0},loss is {1}".format(i, l))

    def run_eval(self, sess, test_x, test_y):
        ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        ds = ds.batch(1)
        X, y = ds.make_one_shot_iterator().get_next()

        # 调用模型
        with tf.variable_scope("model", reuse=True):
            test_prediction, test_loss, test_op = self.lstm_model(test_x, test_y)
        # 预测的数字
        prediction = []
        # 真实的数字
        labels = []
        for i in range(1000):
            pre, l = sess.run([test_prediction, y])
            prediction.append(pre)
            labels.append(l)
        print(prediction)
        print(labels)
