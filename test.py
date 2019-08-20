# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:47:09 2019

@author: JIEYOUNGJUN
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:34:38 2019

@author: JIEYOUNGJUN
"""

import csv
from pylab import *
from scipy.stats import laplace
from scipy.stats import norm

def loaddata(filename):
    data = []
    #with open(filename,'r', decoding='gbk') as f:
    f=open(filename,'r',encoding='gbk')
    reader = csv.reader(f)  # 读取数据
    for row in reader:  # 按行读取
        data.append(row)

    data = np.array(data)  # 转换成矩阵
    #m, n = np.shape(data)  # m:行数    n：列数
    data = data.astype('float64')  # 转换成浮点型
    return data

# 求取均值和方差
def mle_2(x):
    u = np.mean(x)  # 返回均值
    b = math.sqrt(np.mean(pow((x - u), 2)))# 返回方差
    return u, b
    
filename_1 = r'C:\Users\JIEYOUNGJUN\Desktop\determine\LSTMTesterror.csv'
all_info=loaddata(filename_1)
error=all_info[:,2]-all_info[:,1]
test_list=[]
temp1=[]
temp2=[]
temp3=[]
temp4=[]
temp5=[]
temp6=[]
temp7=[]
temp8=[]
temp9=[]
temp10=[]
temp11=[]
test_list.append(temp1)
test_list.append(temp2)
test_list.append(temp3)
test_list.append(temp4)
test_list.append(temp5)
test_list.append(temp6)
test_list.append(temp7)
test_list.append(temp8)
test_list.append(temp9)
test_list.append(temp10)
test_list.append(temp11)
for i in range(4015):
    test_list[i%11].append(error[i])
for i in range(11):
    temp_u,temp_v=mle_2(test_list[i])
    print(str(temp_u)+','+str(temp_v))
#print()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    