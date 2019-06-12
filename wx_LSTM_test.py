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
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 中文图例
mpl.rcParams['axes.unicode_minus'] = False
model = load_model('model.h5')


## 载入数据
#def loaddata(filename):
#    data = []
#    f=open(filename,'r',encoding='gbk')
#    reader = csv.reader(f)  # 读取数据
#    for row in reader:  # 按行读取
#        data.append(row)
#
#    data = np.array(data)  # 转换成矩阵
#    data = data.astype('float64')  # 转换成浮点型
#    feature = data[:, 0:9]  # 前九个作为特征
#    label = data[:, -1]  # 最后一个作为真实值
#    return feature, label

def loaddata(filename):
    data = []
    f=open(filename,'r',encoding='gbk')
    reader = csv.reader(f)  # 读取数据
    for row in reader:  # 按行读取
        data.append(row)

    data = np.array(data)  # 转换成矩阵
    data = data.astype('float64')  # 转换成浮点型
    feature = data[:, 0:12]  # 前九个作为特征
    label = data[:, -1]  # 最后一个作为真实值
    return feature, label


# 处理成模块化数据
def pre(feature, label):
#    Ndays = feature.shape[0] // 11
#    train_x = feature.reshape(Ndays, 11, 9)  # 生成输入数据：三维矩阵（Ndays*11*9）
#    train_t = label.reshape(Ndays, 11, 1)  # 生成对应标签
    Ndays = feature.shape[0] // 11
    train_x = feature.reshape(Ndays, 11, 9)  # 生成输入数据：三维矩阵（Ndays*11*9）
    train_t = label.reshape(Ndays, 11, 1)  # 生成对应标签
    return train_x, train_t


# 计算评估指标
def caculate(y_pre, y_tru):
    # 计算MAPE 平均绝对百分比误差
    v = abs((y_tru-y_pre) / y_tru)
    loss = sum(v) * 100 / len(y_tru)
#    print("MAPE loss", loss)

    # 计算RMSE 均方根误差
    v = mean_squared_error(y_pre, y_tru)
    loss = math.sqrt(sum(v))
    print("the RMSE  is :", loss)

    # 论文中RMSE 计算方法
    # n = len(y_pre)
    # v = 0
    # for x in range(n // 11):
    #     for j in range(11):
    #         v = pow((y_tru-y_pre), 2)
    # loss = sqrt(sum(v) / n)
    # print("the RMSE  is :", loss)

    # 计算MABE 平均绝对偏误差
    loss = mean_absolute_error(y_tru, y_pre)
    print("the MAE of is :", loss)


def main():
    maxNum=0.9799#Need to set
    minNum=0#Need to set
#    filename_3 = "data\pre-data/test_NREL_solar_data.csv"
    #filename_3=r"data\train-data\trainSolar.csv"
    filename_3=r"test.csv"
    feature_3, label_3 = loaddata(filename_3)
    test_x, test_t = pre(feature_3, label_3)
    
#    filename_4 = "data\pre-data/test_NREL_solar_data.csv"
#    #filename_3=r"data\train-data\trainSolar.csv"
#    feature_4, label_4 = loaddata1(filename_4)
##    true_x, true_t = pre(feature_4, label_4)

    y_pre = model.predict(test_x)  # 预测未来辐照度，输出为Ndays_te*11*1的三维矩阵
    y_te = y_pre.reshape(label_3.shape[0], )  # 转成一个一维矩阵
#    y=y_te
#     归一化还原
#    y = 1089.5147 * (y_te + 1) / 2 - 1.6185
#    for i in range(label_3.shape[0]):
#        if y[i] < 0:
#            y[i] = 0
#
#    label = 1089.5147 * (label_3 + 1) / 2 - 1.6185
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
#
#    pyplot.plot(y[:11], label='预测值')
#    pyplot.plot(label[:11], label='真实值')
#    pyplot.legend()
#    pyplot.show()


if __name__ == '__main__':
    main()
