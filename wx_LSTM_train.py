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
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False
K.clear_session()

def loaddata(filename):
    data = []
    with open(filename) as f:
        reader = csv.reader(f) 
        for row in reader: 
            data.append(row)

    data = np.array(data) 
    data = data.astype('float64')  
    feature = data[:, 0:9]  
    label = data[:, -1]  
    return feature, label


# 处理成模块化数据
def pre(feature, label):
    Ndays = feature.shape[0] // 11
    train_x = feature.reshape(Ndays, 11, 9)
    train_t = label.reshape(Ndays, 11, 1) 
    return train_x, train_t


def main():
#    filename_1 = "data\pre-data/train_NREL_solar_data.csv"
    filename_1=r"train.csv"
    feature_1, label_1 = loaddata(filename_1)
    train_x, train_t = pre(feature_1, label_1)

#    filename_2 = "data\pre-data/test_NREL_solar_data.csv"
    filename_2=r"test.csv"
    feature_2, label_2 = loaddata(filename_2)
    validate_x, validate_t = pre(feature_2, label_2)

    model = Sequential()
    model.add(LSTM(60, input_shape=(11, 9), return_sequences=True))
    model.add(Dense(1, activation='linear'))  
    model.compile(loss='mse', optimizer='adam') 

    # 训练模型，迭代100轮，批处理为50
    history = model.fit(train_x, train_t, epochs=300, batch_size=100, validation_data=(validate_x, validate_t))

    model.save('model.h5')
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='validate_loss')
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
