# -*- coding: utf-8 -*-
import model
import sys

sys.path.append('...')
from conf import dnn
import get_data as gt
import tensorflow as tf


class train(object):
    def __init__(self):
        self.train_model()

    def train_model(self):
        k = model.modeler()
        ux = tf.placeholder(shape=(None, 4), dtype=float)  # ux作为绝对正确值
        loss = -tf.reduce_mean(ux * tf.log(tf.clip_by_value(k.model, 0, 1)))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            data = gt.data()
            for i in range(200000):  # 此处可以使用配置文件中的参数
                sess.run(train_step, feed_dict={k.x: data.train_future, ux: data.train_label})
                print(sess.run(loss, feed_dict={k.x: data.train_future, ux: data.train_label}))
            # 训练结束后保存模型
            saver = tf.train.Saver()
            PATH = dnn.CONFIG['train']['save_path']
            saver.save(sess, PATH)
