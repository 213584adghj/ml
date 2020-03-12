# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from numpy import *
import sys
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('...')
from conf import ada_bosting as cnn


class predict():
    def __init__(self):
        self.data = self.load_data()
        self.model = self.load_model()

    def load_model(self):
        return

    def load_data(self):
        return

    def save_result(self):
        return
