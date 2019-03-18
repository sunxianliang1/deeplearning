# coding=utf-8
 
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import math

x=np.random.rand(100,2)
x[50:,1]=x[50:,1]+0.9
x[50:,0]=x[50:,0]+1.0
flag=np.zeros([100,1])
flag[50:]=1


plt.scatter(x[50:,1],x[50:,0],c = 'r')
plt.scatter(x[0:49,1],x[0:49,0],c = 'b')


w=np.zeros([3,1]) #weight
m=int(x.shape[0])
xx=np.ones([m,1])
x=np.concatenate((x,xx),axis = 1)

k=0.1
i=1

while i<5000:
    h=np.dot(x,w)
    g=1/(1+math.e**(-h))
    r=flag-g
    for j in range(3):
        r1=np.vdot(r,x[...,j])
        w[j]=w[j]+k/100*r1
    i=i+1
print(w)
x1=[0,-w[2][0]/w[0][0]]
y1=[-w[2][0]/w[1][0],0]
plt.plot(x1,y1)

plt.show()
