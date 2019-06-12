import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K
from pylab import *
import csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 中文图例
mpl.rcParams['axes.unicode_minus'] = False
K.clear_session()


# 载入数据
def loaddata(filename):
    data = []
    with open(filename) as f:
        reader = csv.reader(f)  # 读取数据
        for row in reader:  # 按行读取
            data.append(row)

    data = np.array(data)  # 转换成矩阵
    data = data.astype('float64')  # 转换成浮点型
    feature = data[:, 0:9]  # 前九个作为特征
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


def main():
#    filename_1 = "data\pre-data/train_NREL_solar_data.csv"
    filename_1=r"train_validate.csv"
    feature_1, label_1 = loaddata(filename_1)
    train_x, train_t = pre(feature_1, label_1)

#    filename_2 = "data\pre-data/test_NREL_solar_data.csv"
    filename_2=r"test.csv"
    feature_2, label_2 = loaddata(filename_2)
    validate_x, validate_t = pre(feature_2, label_2)

    # 搭建LSTM：输入维度（11*9），输出1*50
    # return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
    model = Sequential()
    model.add(LSTM(60, input_shape=(11, 9), return_sequences=True))
    model.add(Dense(1, activation='linear'))  # 搭建全连接层：激活函数：linear，输出维度为1
    model.compile(loss='mse', optimizer='adam')  # 编译模型，指明损失函数和优化方法

    # 训练模型，迭代100轮，批处理为50
    history = model.fit(train_x, train_t, epochs=200, batch_size=50, validation_data=(validate_x, validate_t))

    model.save('model.h5')

    # 画出训练和验证集下的均方误差
    # History类对象包含两个属性，分别为epoch和history
    # history为字典类型，包含val_loss,val_acc,loss,acc四个key值
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='validate_loss')
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
