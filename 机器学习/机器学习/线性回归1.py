# -*- coding: utf-8 -*-
# https://blog.csdn.net/hustqb/article/details/78193544
import numpy as np 
import pandas
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split

data = pandas.read_csv('E:\深度学习\训练集数据\线性回归\Folds5x2_pp.csv',header=0,engine='python')
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
x_train1=x_train.values
x_test1=x_test.values
x_traina=x_train1

m=int(x_traina.shape[0])
xx=np.ones([m,1])
x_traina=np.concatenate((x_traina,xx),axis = 1)

y_train1=y_train.values
y_testa=y_test.values

#A'Ax=A'b  解析法
A=np.dot(x_traina.T,x_traina)
B=np.dot(x_traina.T,y_traina)
theta1=np.linalg.solve(A,B)
print(theta1)

#梯度下降法

theta2=np.zeros([5,1])
k=1e-6 #梯度
i=1
while i<500000:
    h=np.dot(x_traina,theta2)-y_traina
    for j in range(5):
        r=np.vdot(h,x_traina[...,j])
        theta2[j]=theta2[j]-k/m*r
    i=i+1
print(theta2)

