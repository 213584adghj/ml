# coding: utf-8
import _pickle as cPickle
import os
import numpy as np
import sys

CIFAR_DIR = sys.path[1] + '\\' + 'data' + '\\' + 'text' + '\\' + 'cifar-10-batches-py'
print(os.listdir(CIFAR_DIR))


# In[2]:


def load_data(filename):
    """从文件夹读取数据"""
    with open(filename, 'rb') as f:
        data = cPickle.load(f, encoding='latin1')
        return data['data'], data['labels']


class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        #             for item,label in zip(data,labels):
        #                 if label in [0,1]:
        #                     all_data.append(item)
        #                     all_labels.append(label)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('have no more examples')
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all exceptions')
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


class data(object):
    def __init__(self):
        self.train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
        self.test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]
        self.train_data = CifarData(self.train_filenames, True)
        self.test_data = CifarData(self.test_filenames, False)
