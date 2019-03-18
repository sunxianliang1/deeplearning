import numpy as np 
from matplotlib import pyplot as plt 
import os 
import string 
import sys

fp = open("逻辑回归.txt", "r")
X=[]
Y=[]
for line in fp.readlines():
    [x1, x2, y] = line.strip().split()
    X.append([ float(x1), float(x2)])
    Y.append(float(y))
X1=np.asarray(X)
Y1=np.asarray(Y)
i=0
C=[]
for y in Y1:
    if y:
        C.append('r')
    else:
        C.append('b')
        
plt.scatter(X1[...,0],X1[...,1],c=C)
plt.show()













