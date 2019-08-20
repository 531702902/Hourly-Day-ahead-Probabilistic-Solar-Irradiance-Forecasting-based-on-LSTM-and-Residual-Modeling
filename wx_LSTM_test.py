# -*- coding: utf-8 -*-
# @Time : 2018/9/12 15:38
# @Author : "wx"
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model
from pylab import *
import pandas as pd
import csv
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False
model = load_model('model.h5')

def loaddata(filename):
    data = []
    f=open(filename,'r',encoding='gbk')
    reader = csv.reader(f)  
    for row in reader: 
        data.append(row)

    data = np.array(data) 
    data = data.astype('float64')  
    feature = data[:, 0:12] 
    label = data[:, -1]  
    return feature, label

def pre(feature, label):
    Ndays = feature.shape[0] // 11
    train_x = feature.reshape(Ndays, 11, 9)  
    train_t = label.reshape(Ndays, 11, 1) 
    return train_x, train_t


# 计算评估指标
def caculate(y_pre, y_tru):
    v = abs((y_tru-y_pre) / y_tru)
    loss = sum(v) * 100 / len(y_tru)
    v = mean_squared_error(y_pre, y_tru)
    loss = math.sqrt(sum(v))
    print("the RMSE  is :", loss)
    loss = mean_absolute_error(y_tru, y_pre)
    print("the MAE of is :", loss)


def main():
    maxNum=#Need to set
    minNum=#Need to set
    filename_3=r"test.csv"
    feature_3, label_3 = loaddata(filename_3)
    test_x, test_t = pre(feature_3, label_3)

    y_pre = model.predict(test_x)  
    y_te = y_pre.reshape(label_3.shape[0], )  

    y = maxNum * (y_te + 1) / 2
    for i in range(label_3.shape[0]):
        if y[i] < 0:
            y[i] = 0

    label = maxNum * (label_3 + 1) / 2
    caculate(y, label)
    Y_test_long=label
    prediction=y
    v = list(map(lambda x: (abs((x[0] - x[1])) / len(Y_test_long)), zip(Y_test_long, prediction)))
    loss1 = sum(v)
    print("before MAE :", loss1)
    v = list(map(lambda x: (abs((x[0] - x[1]) / x[0]) / len(Y_test_long)), zip(Y_test_long, prediction)))
    loss2 = sum(v)
    print("before MAPE :", loss2)
    mape=0
    sign=0
    for i in range(0,len(prediction)):
        if Y_test_long[i]!=0:
            mape=mape+abs((prediction[i]-Y_test_long[i])/Y_test_long[i])
        else:
            sign=sign+1
    mape=mape/(len(Y_test_long)-sign)
    print("after MAPE :",mape)
    v = list(map(lambda x: ((pow((x[0] - x[1]), 2)) / len(Y_test_long)), zip(Y_test_long, prediction)))
    loss3 = math.sqrt(sum(v))
    print("before RMSE :", loss3)

    data_3 = pd.DataFrame(data=y, columns=['预测值'])
    data_3['真实值'] = label
    data_3.to_csv(r'error.csv')


if __name__ == '__main__':
    main()
