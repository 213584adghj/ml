# -*- coding: utf-8 -*-
import data as da
import numpy as np
from sklearn.naive_bayes import GaussianNB
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import sys

sys.path.append('...')
from conf import text


class train(object):

    def __init__(self, T):
        self.data = da.data(T)
        self.train_sentences, self.train_label, self.test_sentences, self.test_label = self.get_data()
        self.model = self.translate_model()
        self.train_x, self.train_y, self.test_x, self.test_y = self.make_data()
        self.model = self.get_classification_model()
        print('******************************训练集准确率**********************************')
        print(self.model.score(self.train_x, self.train_y))
        pass

    # 生成数据
    def get_data(self):
        x = self.data.features
        y = self.data.labels
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        return train_x, train_y, test_x, test_y

    # 生成词向量模型
    def translate_model(self):
        model = Word2Vec(self.data.features, size=20, min_count=1)  # 参数可调
        model.save(text.CONFIG['train']['model']['Word2Vec_model_save_path'])  # 保存模型
        return model

    # 加工数据,句子转化为向量，字符型标签转化为数字型标签
    def make_data(self):
        a_label_list = self.data.labels
        label_list = []
        for i in a_label_list:
            if (i not in label_list):
                label_list.append(i)
        train_y = []
        test_y = []
        for i in self.test_label:
            test_y.append(label_list.index(i))
        for i in self.train_label:
            train_y.append(label_list.index(i))
        train_x = []
        test_x = []
        for i in self.train_sentences:
            train_x.append(self.model[i])
        for i in self.test_sentences:
            test_x.append(self.model[i])
        final_train_x = []
        final_test_x = []
        for i in train_x:
            iq = np.mean(i, axis=0)
            final_train_x.append(iq)
        for i in test_x:
            iq = np.mean(i, axis=0)
            final_test_x.append(iq)
        return final_train_x, train_y, final_test_x, test_y

    def get_classification_model(self):
        model = GaussianNB()
        model.fit(self.train_x, self.train_y)
        path = text.CONFIG['train']['model']['classification_model_save_path']
        joblib.dump(model, path)
        return model
