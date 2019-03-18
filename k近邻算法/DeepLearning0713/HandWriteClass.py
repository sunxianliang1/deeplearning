# -*- coding: utf-8 -*-
import  numpy
import  matplotlib.image  as  mpimg
from os import listdir
import  operator
import copy
import time

def img2vector(filename):
    img=mpimg.imread(filename)
    vec=img.copy()
    return vec

def GetTrainData():
    filelist=listdir("E:\\深度学习\\训练集数据\\手写字符\\numbers\\train\\")
    TrainNum=len(filelist)
    TrainData=numpy.zeros((TrainNum,16,16))
    Table=numpy.zeros(TrainNum)-1
    for i in range(TrainNum):
        if(filelist[i].find("png")!=-1):
            filename="E:\\深度学习\\训练集数据\\手写字符\\numbers\\train\\"+filelist[i]
            img=mpimg.imread(filename)
            vec=img.copy()
            TrainData[i,:,:]=vec
            Table[i]=int(filelist[i][4:6])-1
    return TrainData,Table  
             
def classfiy(inData,TrainData,Lable):
    i=0
    len=Lable.shape[0]
    dis=numpy.zeros(len)
    TrainData1=copy.deepcopy(TrainData)
    while  i<len and Lable[i]!=-1:
        TrainData1[i,:,:]=TrainData1[i,:,:]-inData
        TrainData1[i,:,:]=TrainData1[i,:,:]**2
        dis[i]=TrainData1[i,:,:].sum()
        i=i+1

    SortedDistIndex=dis.argsort()
    classCount={}
    for i in range(50):
        voteLable=Lable[SortedDistIndex[i]] 
        classCount[voteLable]=classCount.get(voteLable,0)+1  
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


(TrainData1,Table1)=GetTrainData()
filelist=listdir("E:\\深度学习\\训练集数据\\手写字符\\numbers\\test\\")
TrainNum=len(filelist)
all=right=0
ticks1=time.time()
for i in range(TrainNum):
    if(filelist[i].find("png")!=-1):
        filename="E:\\深度学习\\训练集数据\\手写字符\\numbers\\test\\"+filelist[i]
        indata1=img2vector(filename)
        c=classfiy(indata1,TrainData1,Table1)
        d=int(filelist[i][4:6])-1
        #print(filelist[i],"识别为",c,"真实为",d,c==d,'\n')
        all+=1
        if c==d:
            right+=1 
        else :
            print(filelist[i])     
print("总共",all,"张，识别正确",right,"张，正确率为",right/all)
ticks2=time.time()
print(ticks2-ticks1)
#复制一次57.05274844169617s正确率96%
#不复制50.629085063934326s但结果错误