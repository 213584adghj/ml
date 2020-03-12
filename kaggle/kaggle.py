# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:25:43 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('...')
from conf import kaggle as kg
class data(object):
    def __init__(self):
        self.get()
    def get(self):
        train = pd.read_csv(kg.CONFIG['data']['train_path'])
        test = pd.read_csv(kg.CONFIG['data']['test_path'])
        sns.set(context="paper", font="monospace")  # 可视化
        sns.set(style="white")  # 可视化
        f, ax = plt.subplots(figsize=(10, 6))
        train_corr = train.drop('PassengerId', axis=1).corr()
        sns.heatmap(train_corr, ax=ax, vmax=.9, square=True)
        ax.set_xticklabels(train_corr.index, size=15)
        ax.set_yticklabels(train_corr.columns[::-1], size=15)
        ax.set_title('train feature corr', fontsize=20)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        sns.set_style('white')
        sns.distplot(train.Age.fillna(-20), rug=True, color='b', ax=axes[0])
        ax0 = axes[0]
        ax0.set_title('age distribution')
        ax0.set_xlabel('')

        ax1 = axes[1]
        ax1.set_title('age survived distribution')
        k1 = sns.distplot(train[train.Survived == 0].Age.fillna(-20), hist=False, color='r', ax=ax1, label='dead')
        k2 = sns.distplot(train[train.Survived == 1].Age.fillna(-20), hist=False, color='g', ax=ax1, label='alive')
        ax1.set_xlabel('')

        ax1.legend(fontsize=16)
        f, ax = plt.subplots(figsize=(8, 3))
        ax.set_title('Sex Age dist', size=20)
        sns.distplot(train[train.Sex == 'female'].dropna().Age, hist=False, color='pink', label='female')
        sns.distplot(train[train.Sex == 'male'].dropna().Age, hist=False, color='blue', label='male')
        ax.legend(fontsize=15)
        f, ax = plt.subplots(figsize=(8, 3))
        ax.set_title('Pclass Age dist', size=20)
        sns.distplot(train[train.Pclass == 1].dropna().Age, hist=False, color='pink', label='P1')
        sns.distplot(train[train.Pclass == 2].dropna().Age, hist=False, color='blue', label='p2')
        sns.distplot(train[train.Pclass == 3].dropna().Age, hist=False, color='g', label='p3')
        ax.legend(fontsize=15)
        pos = range(0, 6)
        age_list = []
        for Pclass_ in range(1, 4):
            for Survived_ in range(0, 2):
                age_list.append(train[(train.Pclass == Pclass_) & (train.Survived == Survived_)].Age.values)

        fig, axes = plt.subplots(3, 1, figsize=(10, 6))

        i_Pclass = 1

        for ax in axes:
            a = np.nan_to_num(age_list[i_Pclass * 2 - 2])
            b = np.nan_to_num(age_list[i_Pclass * 2 - 1])
            sns.distplot(a, hist=False, ax=ax, label='Pclass:%d ,survived:0' % (i_Pclass), color='r')
            sns.distplot(b, hist=False, ax=ax, label='Pclass:%d ,survived:1' % (i_Pclass), color='g')
            i_Pclass += 1
            ax.set_xlabel('age', size=15)
            ax.legend(fontsize=15)
        print(train.Sex.value_counts())
        print('********************************')
        print(train.groupby('Sex')['Survived'].mean())