# -*- coding: utf-8 -*-
import sys

sys.path.append('...')
from conf import k_means as km
from sklearn.cluster import KMeans


class trainer(object):
    def __init__(self, data):
        self.data = data
        self.model = self.get_result_model()

    def get_result_model(self):
        n_clusters = km.CONFIG['train']['parameter']['n_clusters']
        max_iter = km.CONFIG['train']['parameter']['max_iter']
        n_init = km.CONFIG['train']['parameter']['n_init']
        init = km.CONFIG['train']['parameter']['init']
        copy_x = km.CONFIG['train']['parameter']['copy_x']
        tol = km.CONFIG['train']['parameter']['tol']
        result = KMeans(n_clusters=n_clusters)
        result.fit(self.data)
        return result
