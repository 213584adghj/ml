# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import numpy as np

sys.path.append('...')
from conf import lstm
import data as da
import random


class train(object):
    '''''
    建立模型&训练模型
    '''

    def __init__(self):
        self.model = self.get_model()
        self.data = da.data()
        self.train_model()

    def get_lstm(self):
        num_units = lstm.CONFIG['train']['model']['lstm']['num_units']
        return tf.contrib.rnn.BasicLSTMCell(num_units=num_units)  # 在配置文件中添加lstm的参数

    def get_model(self):
        layer_count = lstm.CONFIG['train']['model']['layer_count']
        cells = []
        for i in range(layer_count):
            cells.append(self.get_lstm())
        return tf.nn.rnn_cell.MultiRNNCell(cells)

    # 训练模型，定义输入输出
    def train_model(self):
        input_x = tf.placeholder(dtype=float, shape=(None, None, len(self.data.train_x[0])))  # 输入向量
        cell = self.get_model()
        output, state = tf.nn.dynamic_rnn(cell, input_x, dtype=tf.float32)
        print(output)
        y = tf.placeholder(dtype=float, shape=(None, None, 1))
        loss = tf.losses.mean_squared_error(output, y)  # 定义损失函数
        train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
        saver = tf.train.Saver()  # 创建一个在训练时的存储对象
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(lstm.CONFIG['train']['learn_count']):
                start = random.randint(0, len(self.data.train_x) - lstm.CONFIG['train']['bitch_size'] *
                                       lstm.CONFIG['train']['time_step'])
                x, y_ = self.get_x_y(start)
                sess.run(train_step, feed_dict={input_x: x, y: y_})
                print(sess.run(loss, feed_dict={input_x: x, y: y_}))
                print(sess.run(output, feed_dict={input_x: x}))
            # 保存模型

    # 使用该函数获得每一次的训练数据
    def get_x_y(self, start):
        bitch_size = lstm.CONFIG['train']['bitch_size']
        time_step = lstm.CONFIG['train']['time_step']
        result_x = self.data.train_x[start:start + bitch_size * time_step]
        result_y = self.data.train_y[start:start + bitch_size * time_step]
        return result_x.reshape((bitch_size, time_step, len(self.data.train_x[0]))), result_y.reshape(
            (bitch_size, time_step, 1))


class sin_train(object):
    def __init__(self):
        x = tf.placeholder(shape=(1, 2, 1), dtype=float)
        a, b = self.get_hidden_layers_output(x)

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            q = [[[1], [2]]]
            print(sess.run(self.prediction(x), feed_dict={x: q}))
        pass

    # 定义lstm层
    def get_lstm_model(self):
        num_units = lstm.CONFIG['sin_train']['model']['hidden_layer']['lstm']['num_units']  # lstm单元的输出维度
        state_is_tuple = lstm.CONFIG['sin_train']['model']['hidden_layer']['lstm']['state_is_tuple']
        forget_bias = lstm.CONFIG['sin_train']['model']['hidden_layer']['lstm']['forget_bias']
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=state_is_tuple,
                                                 forget_bias=forget_bias, reuse=tf.AUTO_REUSE)
        return lstm_cell

    # 定义隐藏层
    def get_hidden_layers(self):
        lstm_cell = self.get_lstm_model()
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 1)  # 默认只有一层的隐藏层
        return cell

    # 获取隐藏层的输出
    def get_hidden_layers_output(self, x):
        cell = self.get_hidden_layers()
        output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return output, _

    # 预测结果
    def prediction(self, x):
        output, _ = self.get_hidden_layers_output(x)
        predictions = tf.contrib.layers.fully_connected(output[:, -1, :], 1, None)  # 不设置激活函数
        return predictions

    # 可视化
    def visualized(self):
        return
