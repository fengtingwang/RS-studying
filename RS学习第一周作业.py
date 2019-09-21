
# coding: utf-8

import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn import linear_model
# 加载数据
digits = load_digits()
data = digits.data
# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)
# 创建LR分类器
lr = linear_model.LogisticRegression()
lr.fit(train_ss_x, train_y)
predict_y=lr.predict(test_ss_x)
print('Ir准确率: %0.4lf' % accuracy_score(test_y, predict_y))





from sklearn.tree import DecisionTreeClassifier
# 加载数据
digits=load_digits()
data = digits.data
#了解数据
print("keys of digits: \n{}".format(digits.keys()))#数据的表头有哪些





print("target names: {}".format(digits['target_names']))#手写数字识别的结果是怎样的
print("Type of data: {}".format(type(digits['data'])))#数据的格式是怎样的
print("Shape of data: {}".format(digits['data'].shape))#样本量有多大，有多少特征





# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)
# 创建CART分类器
CART = DecisionTreeClassifier()
CART.fit(train_ss_x, train_y)
predict_y=CART.predict(test_ss_x)
print('CART准确率: %0.4lf' % accuracy_score(test_y, predict_y))

