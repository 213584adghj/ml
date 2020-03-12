# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class train(object):
    def __init__(self):
        with tf.name_scope('input'):
            # 定义两个placeholder
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y-input')
        self.prediction, self.loss, self.train_step, self.accuracy = self.get_model()
        self.train_model()
        # 命名空间

        pass

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)  # 平均值
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)  # 标准值
            tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
            tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
            tf.summary.histogram('histogram', var)  # 直方图

    def get_model(self):
        with tf.name_scope('layer'):
            # 创建一个简单的神经网络（无隐藏层）
            with tf.name_scope('wights'):
                W = tf.Variable(tf.zeros([784, 10]), name='W')
                self.variable_summaries(W)
            with tf.name_scope('biases'):
                b = tf.Variable(tf.zeros([10]), name='b')
                self.variable_summaries(b)
            with tf.name_scope('wx_plus_b'):
                wx_plus_b = tf.matmul(self.x, W) + b
            with tf.name_scope('softmax'):
                prediction = tf.nn.softmax(wx_plus_b)

        # 二次代价函数
        # loss = tf.reduce_mean(tf.square(y-prediction))
        # 交叉熵
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=prediction))
            tf.summary.scalar('loss', loss)
        with tf.name_scope('train'):
            # 使用梯度下降法
            train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                # 将结果放在一个bool型列表中
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(prediction, 1))
            with tf.name_scope('accuracy'):
                # 求准确率
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)
        return prediction, loss, train_step, accuracy  # 预测，损失函数，训练过程，准确率

    def train_model(self):
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        batch_size = 100
        # 计算一共有多少个批次
        n_batch = mnist.train.num_examples // batch_size
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            writer = tf.summary.FileWriter('logs', sess.graph)
            for epoch in range(51):
                for batch in range(n_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    summary, _ = sess.run([merged, self.train_step], feed_dict={self.x: batch_xs, self.y: batch_ys})

                writer.add_summary(summary, epoch)
                acc = sess.run(self.accuracy, feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels})
                print("Iter" + str(epoch) + ",Testing Accuracy" + str(acc))
        pass


a = train()
