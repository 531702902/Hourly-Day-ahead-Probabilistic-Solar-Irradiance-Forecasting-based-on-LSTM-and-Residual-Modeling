# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:27:24 2018

@author: JIEYOUNGJUN
"""

import csv
from pylab import *
from scipy.stats import laplace
from scipy.stats import norm
import numpy as np
mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False
standard=0.8
import copy

def loaddata(filename):
    data = []
    f=open(filename,'r',encoding='gbk')
    reader = csv.reader(f)  
    for row in reader: 
        data.append(row)

    data = np.array(data) 
    m, n = np.shape(data)  
    data = data.astype('float64') 
    return data, m, n

def mle_1(x):
    u = np.mean(x) 
    b = np.mean(abs(x - u)) 
    return u, b

def mle_2(x):
    u = np.mean(x)  
    b = np.mean(pow((x - u), 2))
    return u, b


# 产生拉普拉斯置信区间
def lapulasi(loc, scale):
    low, up = laplace.interval(standard)  
    p_low = low * scale + loc
    p_up = up * scale + loc
    return p_low, p_up

def normal(loc, scale):
    scale = sqrt(scale)
    low, up = norm.interval(standard)
    p_low = low * scale + loc
    p_up = up * scale + loc
    return p_low, p_up

def show(error):
    count, bins, ignored = plt.hist(error, bins=500, normed=False)
    pro = count / np.sum(count) 
    plt.plot(bins[:500], count, "r") 
    plt.xlabel("误差")
    plt.ylabel("频数")
    plt.title("频数统计直方图")
    plt.show()
    plt.plot(bins[:500], pro, "r", lw=2)  # 画出频率折线图
    plt.xlabel("误差")
    plt.ylabel("频率")
    plt.title("频率统计直方图")
    plt.show()


def calculate(data, p_low, p_up):
    count = 0
    preNum=1
    truNum=2
    for i in range(len(data)):
        if p_up[i] >= data[i][truNum] and data[i][truNum]>= p_low[i]:
            count = count + 1
    PICP = count / len(data)

    max0 = np.max(data[:, preNum])
    min0 = np.min(data[:, preNum])
    sum0 = p_up - p_low
    sum1 = np.sum(sum0) / len(sum0)
    PINAW = 1 / (max0 - min0) * sum1

    g = 90  
    u = standard
    e0 = math.exp(-g * (PICP - u))

    if PICP >= u:
        r = 0
    else:
        r = 1
    CWC = PINAW * (1 + r * PICP * e0)
#    print("CWC", CWC)
    return count,PINAW,CWC


def main():
    filename_1 = r'LSTMTrainerror.csv'
    filename_2 = r'LSTMTesterror.csv'
    data_1, m_1, n_1 = loaddata(filename_1)
    data_2, m_2, n_2=  loaddata(filename_2)
    preNum=1
    truNum=2
    totalDay=365*7+366*2
    testDay=365
    timeClass=11
    error = data_1[:, truNum] - data_1[:, preNum]
    show(error)

    dataClass=[]
    errorClass=[]
    dataTemp=[]
    errorTemp=[]
    dataTest=[]
    for i in range(timeClass):
        for j in range(totalDay):
            error_temp=data_1[j*11+i, truNum] - data_1[j*11+i, preNum]
            data_temp=data_1[j*11+i]
            dataTemp.append(data_temp)
            errorTemp.append(error_temp)
        c=copy.deepcopy(dataTemp)
        d=copy.deepcopy(errorTemp)
        dataClass.append(c)
        errorClass.append(d)
        dataTemp.clear()
        errorTemp.clear()
        for j in range(testDay):
            data_temp=data_2[j*11+i]
            dataTemp.append(data_temp)
        c=copy.deepcopy(dataTemp)
        dataTest.append(c)
        dataTemp.clear()
    locMerge=[]
    scaleMerge=[]
    lowMerge=[]
    upMerge=[]
    picp=0
    count=0
    pinaw=0
    cwc=0
    for i in range(timeClass):
        loc,scale=mle_1(errorClass[i])
        low,up=lapulasi(loc,scale)
        locMerge.append(loc)
        scaleMerge.append(scale)
        lowMerge.append(low)
        upMerge.append(up)
        data1=np.reshape(dataTest[i],(testDay,3))
        p_low_1 = data1[:, preNum] + low
        p_up_1 = data1[:, preNum] + up
        if len(data1)!=0:
            picptemp,pinawtemp,cwctemp=calculate(data1, p_low_1, p_up_1)
            picp+=picptemp
            pinaw+=pinawtemp
            cwc+=cwctemp
            count=count+1
    print("final picp"+str(picp/timeClass/testDay))
    print("final pinaw"+str(pinaw/count))
    g = 90 
    u = standard
    picp=picp/timeClass/testDay
    pinaw=pinaw/count
    e0 = math.exp(-g * (picp - u))

    if picp >= u:
        r = 0
    else:
        r = 1
    CWC = pinaw * (1 + r * picp * e0)
    print("final cwc"+str(cwc/count))
    print("calculate cwc"+str(CWC))
    
    loc_1, scale_1 = mle_1(error)
    low_1, up_1 = lapulasi(loc_1, scale_1)
    print(loc_1,scale_1)
    print('误差的拉普拉斯',standard*100,'%置信区间：', [low_1, up_1])
    p_low_1 = data_2[:, preNum] + low_1
    p_up_1 = data_2[:, preNum] + up_1
    calculate(data_2, p_low_1, p_up_1)
    
    picp=0
    count=0
    pinaw=0
    cwc=0
    for i in range(timeClass):
        loc,scale=mle_2(errorClass[i])
        low,up=normal(loc,scale)
        data1=np.reshape(dataTest[i],(testDay,3))
        p_low_1 = data1[:, preNum] + low
        p_up_1 = data1[:, preNum] + up
        if len(data1)!=0:
            picptemp,pinawtemp,cwctemp=calculate(data1, p_low_1, p_up_1)
            picp+=picptemp
            pinaw+=pinawtemp
            cwc+=cwctemp
            count=count+1
    print("final picp"+str(picp/timeClass/testDay))
    print("final pinaw"+str(pinaw/count))
    g = 90  
    u = standard
    picp=picp/timeClass/testDay
    pinaw=pinaw/count
    e0 = math.exp(-g * (picp - u))

    if picp >= u:
        r = 0
    else:
        r = 1
    CWC = pinaw * (1 + r * picp * e0)
    print("final cwc"+str(cwc/count))
    print("calculate cwc"+str(CWC))
    
    loc_2, scale_2 = mle_2(error)
    low_2, up_2 = normal(loc_2, scale_2)
    print('误差的高斯',standard*100,'%置信区间：', [low_2, up_2])
    p_low_2 = data_2[:, preNum] + low_2
    p_up_2 = data_2[:, preNum] + up_2
    calculate(data_2, p_low_2, p_up_2)


if __name__ == '__main__':
    main()
