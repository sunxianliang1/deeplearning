# -*- coding: utf-8 -*-
import  numpy
import  matplotlib.image  as  mpimg
from os import listdir
import  operator
filelist=listdir("E:\\深度学习\\训练集数据\\手写字符\\numbers\\train\\")
m=len(filelist)
Vector=numpy.zeros((16,16))
k=0
for i in range(m):
    if(filelist[i].find("png")!=-1):
        filename="E:\\深度学习\\训练集数据\\手写字符\\numbers\\train\\"+filelist[i]
        img=mpimg.imread(filename)
        vec=img.copy()
        Vector=Vector+vec
        k=k+1
Vector=Vector/k
mpimg.imsave("ave.png",Vector,cmap='gray_r')
print(Vector)