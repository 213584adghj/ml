# -*- coding: utf-8 -*-
import sys
from sklearn.model_selection import train_test_split
import jieba.analyse
import xlwt

sys.path.append('...')
from conf import text
import xlrd
import jieba
import re


# base_data = xlrd.open_workbook(text.CONFIG['data']['base_name'])
class data(object):
    # T控制提取数据的数量
    def __init__(self, T):
        self.base_data_name = text.CONFIG['data']['base_name']
        self.stop_word_name = text.CONFIG['data']['text_processing']['stop_word_file']
        self.stop_word_list = self.get_stop_word_tables()
        self.base_data = xlrd.open_workbook(self.base_data_name)
        self.labels = self.get_labels(T)  # 提取标签列表
        # print(self.labels)
        self.features = self.get_features(T)  # 提取特征列表\

    def get_labels(self, T):
        result = []
        labels_sheet = self.base_data.sheet_by_index(-1)
        for i in range(labels_sheet.nrows):
            result.append(str(labels_sheet.row(i)[-1]))
            if (i == T):
                break
        result.remove(result[0])
        return result

    def get_features(self, T):
        result = []
        labels_sheet = self.base_data.sheet_by_index(0)
        for i in range(1, labels_sheet.nrows):
            a = (labels_sheet.row(i)[-1])
            pattern = re.compile(r'[^\u4e00-\u9fa5]')
            chinese = re.sub(pattern, '', a.value)
            result.append(self.text_processing(chinese))
            if (i == T):
                break
        return result

    # 获取停用词表
    def get_stop_word_tables(self):
        stopwords = [line.strip() for line in open(self.stop_word_name, 'r', encoding='utf-8').readlines()]
        # print(stopwords)
        return stopwords

    # 实现分词，去除停用词，抽取关键词的功能
    def text_processing(self, sentence):
        # 分词
        cut_all = text.CONFIG['data']['text_processing']['cut_all']
        list = jieba.lcut(sentence, cut_all=cut_all)
        # 去除停用词
        for i in list:
            if (i in self.stop_word_list):
                list.remove(i)
        # 抽取关键词
        sen = " ".join(list)
        topK = text.CONFIG['data']['text_processing']['topK']
        q = jieba.analyse.extract_tags(sen, topK=topK, withWeight=False, allowPOS=())
        return q

    # 生成训练测试集
    def get_train_test(self):
        test_size = text.CONFIG['data']['random_split']['test_size']
        result = train_test_split(self.features, self.labels, test_size=test_size)
        train_file = self.base_data_name = text.CONFIG['data']['train_file']
        test_file = self.base_data_name = text.CONFIG['data']['test_file']
        return result

    # 将数据写入文件夹
    def write(self, name):
        return
