import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as mp
import sys
from loaddata   import ECTdata
import time


class Traditional:
    def __init__(self,t='tri',sample=20000): 
        self.mydata=ECTdata('E:\deeplearning\ECT\数据生成\data',sample)
        self.mydata.initsca(t=t)
        self.sample=sample
        print("data init success!")
        #print(self.mydata.lmc.shape)   28,702
        self.Srow=np.sum(self.mydata.lmc,axis=0)#702
        self.Scol=np.sum(self.mydata.lmc,axis=1)#28
        self.SLBP=np.zeros([self.mydata.capsize,self.mydata.imgsize])
        self.SLAND=np.zeros([self.mydata.capsize,self.mydata.imgsize])
        for i in range(self.mydata.imgsize):
            for j in range(self.mydata.capsize):
                self.SLBP[j][i]=self.mydata.lmc[j][i]/self.Srow[i]
        self.Srow2=np.sum(self.SLBP,axis=1)
        for i in range (self.mydata.capsize):
            for j in range(self.mydata.imgsize):
                self.SLAND[i][j]=self.SLBP[i][j]/self.Srow2[i]
        self.yLBP=np.zeros([self.sample,self.mydata.imgsize])
        self.yLAND=np.zeros([self.sample,self.mydata.imgsize])
       

    def LBP(self):
       for i in range(18900,19100):
           self.yLBP[i]=np.matmul(self.SLAND.T,self.mydata.caps[i].reshape(-1,1)).reshape(-1)*self.mydata.imgsize/2
           
    
    def LAND(self):
        X= np.matrix(np.matmul(self.SLAND.T,self.SLAND))
        e, v = np.linalg.eig(X)
        ee=np.amax(e)
        b=2/ee
        a=b.real
        start=time.clock()
        for i in range(18900,19100):
            g=np.matmul(self.SLAND.T,self.mydata.caps[i].reshape(-1,1)).reshape(-1)*self.mydata.imgsize/28
            for k in range(100):
                gk=g-a*np.dot(self.SLAND.T,np.dot(self.SLAND,g)-self.mydata.caps[i]*2)
                #gk=g-a*np.dot(self.SLAND.T,np.dot(self.SLAND,g)-self.mydata.empty.reshape(-1)*2)
                gk[gk>1]=1
                gk[gk<0]=0
                g=gk
            self.yLAND[i]=g
        end=time.clock()
        print("重建使用时间为：%f"%(end-start))

    def printpic(self,index,t='fig'):      
        self.mydata.drawsca(self.mydata.images[index],t=t)

        self.yLBP[self.yLBP>1]=1
        self.yLBP[self.yLBP<0]=0
        self.mydata.drawsca(self.yLBP[index],t=t)

        self.yLAND[self.yLAND>1]=1
        self.yLAND[self.yLAND<0]=0
        self.mydata.drawsca(self.yLAND[index],t=t)
 
        s=np.std(self.yLBP[index]-self.mydata.images[index])
        print("LBP标准差为%f"%s)
        s=np.std(self.yLAND[index]-self.mydata.images[index])
        print("LAND标准差为%f"%s)
        mp.show()

if __name__=='__main__':
    trad=Traditional(t='tri')
    trad.LBP()
    trad.LAND()
    i=0
    while 1:

        i=input("输入图片序号")
        i=int(i)
        if i<0:
            break
        trad.printpic(i,t='tri')
        



