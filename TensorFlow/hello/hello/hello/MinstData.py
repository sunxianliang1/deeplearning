#coding:utf-8

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import random

class MinstData:
    def __init__(self,path,type='zip'):
        if type=='zip':
            self.TrainImg,self.TrainLabel,self.TrainLen=self.LoadMnist1(path)     #训练集
            self.TestImg,self.TestLabel,self.TestLen=self.LoadMnist1(path,"t10k") #测试集
            self.size=[28,28]
        else:
            self.TrainImg,self.TrainLabel,self.TrainLen=self.LoadMnist2(path)     #训练集
            self.TestImg,self.TestLabel,self.TestLen=self.LoadMnist2(path,"test") #测试集
            self.size=[16,16]

        self.TrainImg=self.TrainImg/256
        self.TestImg=self.TestImg/256
        self.TrainLabel10=np.zeros([self.TrainLen,10])
        for i in range(self.TrainLen):
            self.TrainLabel10[i][self.TrainLabel[i]]=1
        self.TestLabel10=np.zeros([self.TestLen,10])
        for i in range(self.TestLen):
            self.TestLabel10[i][self.TestLabel[i]]=1

    def LoadMnist1(self,path, kind='train'):#压缩版
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
        images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            labels = np.fromfile(lbpath,dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        lens=len(labels)
        return images, labels,lens

    def LoadMnist2(self,path, kind='train'):#图片版
        """Load MNIST data from `path`"""
        filelist=listdir(path+"\\"+kind+"\\")
        lens=len(filelist)
        images=np.zeros([lens,16*16])
        labels=np.zeros([lens],dtype=np.int)-1
        for i in range(lens):
            if(filelist[i].find("png")!=-1):
                filename=path+"\\"+kind+"\\"+filelist[i]
                img=mpimg.imread(filename)
                vec=img.copy()
                images[i,:]=vec.reshape(16*16)
                labels[i]=int(filelist[i][4:6])-1
        np.random.seed(100)#随机种子，但两次必须相同
        np.random.shuffle(images)
        np.random.seed(100)
        np.random.shuffle(labels)

        return images, labels,lens
    def PrintSomeImg(self):
        fig, ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True, )
        ax = ax.flatten()
        for i in range(10):
            img = self.TrainImg[self.TrainLabel == i][0].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        plt.tight_layout()
        plt.show()
    
    def PrintImg(self,index):
        img = self.TrainImg[index].reshape(28, 28)
        plt.imshow(img,cmap='Greys')
        plt.show()
    
    def next_batch(self,num):
        selectlist=random.sample(range(0,self.TrainLen-1),num)
        index=np.asarray(selectlist)
        x=self.TrainImg[index]
        y=np.zeros([num,10])
        for i in range(num):
            y[i][self.TrainLabel[index[i]]]=1  
        return x,y


if __name__=='__main__':
    minst=MinstData("E:\深度学习\训练集数据\手写字符\压缩版")
    minst.PrintImg(100)
