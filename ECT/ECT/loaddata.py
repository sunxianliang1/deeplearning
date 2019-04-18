import scipy.io
import os
import glob
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm as cm
import matplotlib.colors as col
import csv
import re

class ECTdata:
    def __init__(self,path,size=0):
        empty1=scipy.io.loadmat(path+'\empty.mat')
        efull1=scipy.io.loadmat(path+'\efull.mat')
        self.empty=np.asarray(empty1['Cap'])    #空管标定数据
        self.efull=np.asarray(efull1['Cap'])    #满管标定数据
        self.capsub=np.subtract(self.efull,self.empty)#空满管差值        
        paths=glob.glob(path+'\data*.mat')
        paths.sort(key=lambda x:int(re.search('\d+',x).group(0)))
        if size==0:
            size=len(paths)

        data=scipy.io.loadmat(paths[0])
        self.imgsize= data['data'][0][0][5].size    #图片数组长度
        self.capsize= data['data'][0][0][6].size    #电容数组长度
        self.images=np.zeros([size,self.imgsize])
        self.caps=np.zeros([size,self.capsize])
        self.imgsub=np.ones([1,self.imgsize])*3
        self.imgempty=np.ones([1,self.imgsize])
        #不要用append，numpy不支持动态分配，append每次都会将原数组复制，O（n^2)
        #预先分配空间，O(n) 
        for file in paths:
            ff=re.search(r'\d+',file).group(0)
            ff=int(ff)

            data=scipy.io.loadmat(file)
            a=data['data'][0][0][5]
            b=np.divide(np.subtract(a,self.imgempty),self.imgsub)#图片归一化
            self.images[ff-1]=b[0]

            c=data['data'][0][0][6]
            d=np.divide(np.subtract(c,self.empty),self.capsub)  #电容归一化
            self.caps[ff-1]=d[0]
            
            if ff%100==99:
                print(ff)
            if ff==size:
                break

        s=int(size*0.95)
        self.imgtrain,self.imgtest=np.split(self.images,[s])
        self.captrain,self.captest=np.split(self.caps,[s])

    def initsca(self,t='fig'):
        startcolor = '#005EFF'   
        midcolor = '#2CFF4B'    
        endcolor = '#F43931'         
        cmap2 = col.LinearSegmentedColormap.from_list('own2',[startcolor,midcolor,endcolor]) 
        cm.register_cmap(cmap=cmap2)
        if t=='fig':
            self.point=np.zeros([200,200,6,2])
            csvfile=csv.reader(open('draw.csv','r'))
            m=True
            for line in csvfile:
                if m:
                    i=int(line[0])-1
                    j=int(line[1])-1
                    m=False
                else:
                    m=True
                    for k in range(6):
                        self.point[i][j][k][0]=int(line[k*2])-1
                        self.point[i][j][k][1]=float(line[k*2+1])
        if t=='tri':
            csvfile=csv.reader(open('draw1.csv','r'))
            self.point=np.zeros([200,200])
            for line in csvfile:
                i=int(line[0])-1
                j=int(line[1])-1
                self.point[i][j]=j=int(line[2])-1

    def drawsca(self,image,t='fig'):
        point1=np.ones([200,200])*0.5
        if t=='fig':
            for i in range(200):
                for j in range(200):
                    if (i-100)**2+(j-100)**2<10000:
                        point1[i][j]=0
                        for k in range(6):
                            point1[i][j]=point1[i][j]+image[int(self.point[i][j][k][0])]*self.point[i][j][k][1]     
        if t=='tri':
            for i in range(200):
                for j in range(200):
                    if (i-100)**2+(j-100)**2<10000:  
                        point1[i][j]=1            
                    if (i-100)**2+(j-100)**2<9850:
                        point1[i][j]=image[int(self.point[i][j])]
                                 
        mp.figure()        
        mp.imshow(point1,cmap='own2')
        mp.colorbar()
        




