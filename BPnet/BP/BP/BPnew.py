#coding:utf-8

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import copy

class BPnet:
    def __init__(self,path,rate=0.010):
        self.TrainImg,self.TrainLabel,self.TrainLen=self.LoadMnist(path)     #训练集
        self.TestImg,self.TestLabel,self.TestLen=self.LoadMnist(path,"t10k") #测试集
        self.LayerLen={1:784,2:86,3:10}
        self.LayerDim=3
        self.w={} #权值
        self.b={} #偏差
        self.a={} #结果
        self.z={} #输出
        self.delta={} #偏差对z的导数
        self.dZ={} #sigmod函数关于z的导数
        self.out=np.zeros([1,10])#样本输出
        self.rate=rate#学习速率
        self.TrainNum=60000
        self.begin=0
        init=0.1
        for i in range(2,self.LayerDim+1):
            self.w[i]=np.random.randn(self.LayerLen[i-1],self.LayerLen[i])/np.sqrt(self.LayerLen[i-1])*init
            self.b[i]=np.zeros([1,self.LayerLen[i]])
            self.a[i]=np.zeros([1,self.LayerLen[i]])
            self.z[i]=np.zeros([1,self.LayerLen[i]])
            self.dZ[i]=np.zeros([1,self.LayerLen[i]])
            self.delta[i]=np.zeros([1,self.LayerLen[i]])
        fo = open("三层压缩new.txt", "a")
        print("rate=%f,LayerDim=%d  init=%f"%(rate,self.LayerDim,init),file=fo)
        print(self.LayerLen,file=fo)
        print("Train samples is %d"%(self.TrainNum),file=fo)
        fo.close()

    def Train(self,MaxIterNum=100,MaxErr=0.1):
        for TrainIter in range(1,MaxIterNum+1):
            for SampleIter in range(self.begin,self.TrainNum+self.begin):
                #self.PrintImg(SampleIter)
                self.a[1]=copy.deepcopy(self.TrainImg[SampleIter])/256
                self.a[1].shape=(1,-1)#不能用reshape，reshape不改变本身，值改变返回值
                self.out=np.zeros([1,10])
                self.out[0,self.TrainLabel[SampleIter]]=1
                lasterr=10000
                count=0
                while True:
                    self.ForwardTransfer()
                    Err=self.GetErr(SampleIter)
                    if Err<MaxErr  or count>100or abs(lasterr-Err)<0.000001: # 
                        break
                    lasterr=Err
                    self.BackwardTransfer()
                    count=count+1
                #print("The %d th picture use %d counts tor train.\n" %(SampleIter+1,count))
            correct,err1,err2=self.GetAccur()
            print("This is the %d th trainning NetWork!The correct rate is %f%%,err1 is %f%%,err2 is %f%%" %(TrainIter,correct,err1,err2))
            if correct>95:
                break;   
        fo = open("三层压缩.txt", "a")
        print("In trainning %d times NetWork,The correct rate is %f%%,err1 is %f%%,err2 is %f%%" %(TrainIter,correct,err1,err2),file=fo)
        fo.close()  

    def Test(self):
        correct=0
        err1=0
        err2=0
        num=self.TrainNum//6
        for SampleIter in range(num):
            self.a[1]=copy.deepcopy(self.TestImg[SampleIter])/256
            self.a[1].shape=(1,-1)#不能用reshape，reshape不改变本身，值改变返回值
            self.out=np.zeros([1,10])
            self.out[0,self.TestLabel[SampleIter]]=1
            self.ForwardTransfer()
            err=0
            for i in range(10):
                if abs(self.a[3][0][i]-self.out[0,i])>0.33:
                    err=err+1
            if err==0:        
                correct+=1
            elif err==1:
                err1+=1
            else:
                err2+=1
        print("In Test NetWork!The correct rate is %f%%,err1 is %f%%,err2 is %f%%" %(correct/num*100,err1/num*100,err2/num*100))
        fo = open("三层压缩.txt", "a")
        print("In Test NetWork!The correct rate is %f%%,err1 is %f%%,err2 is %f%%" %(correct/num*100,err1/num*100,err2/num*100),file=fo)
        print("---------------------------------------------------------------------\n",file=fo)
        fo.close()

    def GetAccur(self):
        correct=0
        err1=0
        err2=0
        for SampleIter in range(self.begin,self.TrainNum+self.begin):
            self.a[1]=copy.deepcopy(self.TrainImg[SampleIter])/256
            self.a[1].shape=(1,-1)#不能用reshape，reshape不改变本身，值改变返回值
            self.out=np.zeros([1,10])
            self.out[0,self.TrainLabel[SampleIter]]=1
            self.ForwardTransfer()
            err=0
            for i in range(10):
                if abs(self.a[3][0][i]-self.out[0,i])>0.33:
                    err=err+1
            if err==0:        
                correct+=1
            elif err==1:
                err1+=1
            else:
                err2+=1
        return correct/self.TrainNum*100,err1/self.TrainNum*100,err2/self.TrainNum*100

    def ForwardTransfer(self):
        for i in range(2,self.LayerDim+1):
            self.z[i]=np.dot(self.a[i-1],self.w[i])+self.b[i]
            #self.a[i]=self.sigmod(self.z[i])
            self.a[i]=self.relu(self.z[i])

    def GetErr(self,SampleIter):
        err=0
        for i in range(10):
            err=err+(self.a[3][0][i]-self.out[0,i])**2
        return err/2

    def BackwardTransfer(self):
        for i in range(2,self.LayerDim+1):
            #self.dZ[i]=self.z[i]*(1-self.z[i])
            self.dZ[i]=np.array(self.z[i]>0)*1.0
        self.delta[self.LayerDim]=np.multiply(self.a[self.LayerDim]-self.out,self.dZ[self.LayerDim])
        for i in range(2,self.LayerDim):
            self.delta[i]=np.multiply(np.dot(self.delta[i+1],self.w[i+1].transpose((1,0))),self.dZ[i])
        dEdW={}
        dEdb={}
        for i in range(2,self.LayerDim+1):
            #这里不能用.T，会将1*n的矩阵变为(n,)的矩阵，维度会减一
            dEdW[i]=np.dot(self.a[i-1].transpose((1,0)),self.delta[i])
            dEdb[i]=self.delta[i]
        #updata w and b
        for i in range(2,self.LayerDim+1):
            self.w[i]=self.w[i]-self.rate*dEdW[i]
            self.b[i]=self.b[i]-self.rate*dEdb[i]

    def LoadMnist(self,path, kind='train'):
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

    def sigmod(self,z):
        """激活函数sigmod"""
        return 1/(1 + np.exp(-z))
    def relu(self,z):
        """激活函数relu"""
        return np.array(z>0)*z
    
    
        

if __name__=='__main__':
    BP=BPnet("E:\深度学习\训练集数据\手写字符\压缩版")
    BP.Train()
    BP.Test()