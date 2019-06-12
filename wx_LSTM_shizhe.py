# -*- coding: utf-8 -*-
# @Time : 2018/9/29 19:43
# @Author : "wx"
import csv
import glob
from keras import backend as K
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from matplotlib import pyplot
from pylab import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 中文图例
mpl.rcParams['axes.unicode_minus'] = False
K.clear_session()


# 载入数据
def loaddata(filename):
    data_1 = []
    print(filename)
    with open(filename) as f:
        reader = csv.reader(f)  # 读取数据
        for row in reader:  # 按行读取
            data_1.append(row)
    return data_1


def hebing():
    count = 0
    data_2 = []
    csv_list = glob.glob('data/train-data\*.csv')
    for i in csv_list:
        data_1 = loaddata(i)
        data_2.append(data_1)
        count = count + 1
    #print(data_2[5])
    return data_2


# 划分数据集
def huafen(data_2, i):
    validate = data_2[i]
    train = []
    for j in range(len(data_2)):
        if j != i:
            train = data_2[j] + train
    train = np.array(train)
    validate = np.array(validate)

    feature_t = train[:, 0:9]  # 前九个作为特征
    label_t = train[:, -1]  # 最后一个作为真实值

    feature_v = validate[:, 0:9]  # 前九个作为特征
    label_v = validate[:, -1]  # 最后一个作为真实值
    return feature_t, label_t, feature_v, label_v


# 处理成模块化数据
def pre(feature, label):
    Ndays = feature.shape[0] // 11
    train_x = feature.reshape(Ndays, 11, 9)  # 生成输入数据：三维矩阵（Ndays*11*9）
    train_t = label.reshape(Ndays, 11, 1)  # 生成对应标签
    return train_x, train_t


def main():
    data_2 = hebing()
    loss_t = np.zeros(10)
    loss_v = np.zeros(10)

    for i in range(10):
        feature_t, label_t, feature_v, label_v = huafen(data_2, i)
        train_x, train_t = pre(feature_t, label_t)
        validate_x, validate_t = pre(feature_v, label_v)

        # 搭建LSTM：输入维度（11*9），输出1*50
        # return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
        model = Sequential()
        model.add(LSTM(40, input_shape=(11, 9), return_sequences=True))
        model.add(Dense(1, activation='linear'))  # 搭建全连接层：激活函数：linear，输出维度为1
        model.compile(loss='mse', optimizer='adam')  # 编译模型，指明损失函数和优化方法

        # 训练模型，迭代100轮，批处理为50
        history = model.fit(train_x, train_t, epochs=100, batch_size=50, validation_data=(validate_x, validate_t))
        model.save('model-%s.h5' % (i+1))

        # 画出训练和验证集下的均方误差
        # History类对象包含两个属性，分别为epoch和history
        # history为字典类型，包含val_loss,val_acc,loss,acc四个key值
        pyplot.plot(history.history['loss'], label='train_loss')
        pyplot.plot(history.history['val_loss'], label='validate_loss')
        pyplot.legend()
        pyplot.show()

        a = history.history['loss']
        b = history.history['val_loss']
        loss_t[i] = min(a)
        loss_v[i] = min(b)

    print(loss_t)
    print(loss_v)


if __name__ == '__main__':
    main()
