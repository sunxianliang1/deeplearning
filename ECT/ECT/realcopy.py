import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
from loaddata   import ECTdata
import tensorflow
from tensorflow.keras.models import load_model
from matplotlib import animation

class realplay():
    def __init__(self):
        # if 'session' in locals() and tensorflow.session is not None:
        #     print('Close interactive session')
        #     tensorflow.session.close()
        # config = tensorflow.ConfigProto()
        # config.gpu_options.allow_growth = True  #允许显存增长
        # tensorflow.keras.backend.set_session(tensorflow.Session(config=config))
        # print('GPU memory is allowed to growth.')
        # tensorflow.keras.backend.clear_session()

        self.path="E:\deeplearning\程序\ECT\ECT\真实实验";
        empty1=scipy.io.loadmat(self.path+'\empty.mat')
        efull1=scipy.io.loadmat(self.path+'\efull.mat')
        lmc1=scipy.io.loadmat(self.path+'\lmc.mat')
        self.fltempty=np.asarray(empty1['Cap'])    #空管标定数据
        self.fltfull=np.asarray(efull1['Cap'])    #满管标定数据
        self.lmc  =np.asarray(lmc1['S'])       #灵敏场
        self.path="E:\会议项目\第九届流态化会议\处理数据\加料高度\\v15l168\\2016-12-13_13-04-28.203"
        with open(self.path+'\calibration.txt','r') as f:
            lines=f.readlines()
            strempty=lines[1].split()
            intempty=list(map(int,strempty))
            self.intempty=np.asarray(intempty)
            strfull=lines[3].split()
            intfull=list(map(int,strfull))
            self.intfull=np.asarray(intfull)
        with open("E:\会议项目\第九届流态化会议\处理数据\\app\\option\\calibration\\efull.txt",'r') as f:
            l=f.readline()
            strfull=l.split()
            fltfull=list(map(float,strfull))
            self.oldfltfull=np.asarray(fltfull)
        with open("E:\会议项目\第九届流态化会议\处理数据\\app\\option\\calibration\\empty.txt",'r') as f:
            l=f.readline()
            strempty=l.split()
            fltempty=list(map(float,strempty))
            self.oldfltempty=np.asarray(fltempty)    

        self.intdelt=self.intfull-self.intempty
        self.fltdelt=self.fltfull-self.fltempty
        self.oldfltdelt=self.oldfltfull-self.oldfltempty
        index=np.argmax(self.intdelt)
        self.k=self.intdelt[index]/self.fltdelt[0][index]
        self.k2=self.intdelt[index]/self.oldfltdelt[index]
        print (self.k)
        print (self.k2)
        self.draw=ECTdata('E:\deeplearning\ECT\数据生成\data',size=200)
        self.draw.initsca(t='tri')
        self.dn=DN()
        self.land=Land()


    def play(self):
        with open(self.path+'\cdata1.txt','r') as f:
            self.lines=f.readlines()
            self.lineindex=0

        fig, ax = plt.subplots(1,2)
        self.ax1 = ax[0]
        self.ax2 = ax[1]

        initarr=np.ones([200,200])
        line1=self.ax1.imshow(initarr,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        line2=self.ax2.imshow(initarr,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)

        self.index=1
        self.lineindex=300
        #frames 动画长度，一次循环包含的帧数，在函数运行时，其值会传递给函数update(n)的形参“n”
        ani = animation.FuncAnimation(fig, self.calculate, frames=self.index, init_func=self.initpic, blit=True)
        #plt.colorbar()
        ax[0].axis('off')
        ax[1].axis('off')
        #fig.colorbar(line2,ax=[ax[0],ax[1]])
        cax = plt.axes([0.92, 0.25, 0.025, 0.48])
        fig.colorbar(line2,cax=cax)
        plt.show()

    def play2(self):
        with open(self.path+'\cdata1.txt','r') as f:
            self.lines=f.readlines()
            self.lineindex=0
        
        a=self.lines[3].split(' ',1)
        strmeasure=a[1].split()
        fltmeasure=list(map(float,strmeasure))
        measure=np.asarray(fltmeasure)
        print (a[0])

        de=(fltmeasure-self.oldfltempty)*self.k2

        fltmeasure=de/self.k+self.fltempty

        d=np.divide(np.subtract(fltmeasure,self.fltempty),self.fltdelt)  #电容归一化

        # ydn=self.dn.caul(d).T.reshape(-1)
        yland=self.land.caul(d).reshape(-1)
        t='tri'
        # drawdn=self.draw.drawdata(ydn,t=t)
        drawland=self.draw.drawdata(yland,t=t)

        fig, ax = plt.subplots(1,2)
        self.ax1 = ax[0]
        self.ax2 = ax[1]
        
        # line1=self.ax1.imshow(drawdn,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        line1=self.ax1.imshow(drawland,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        line2=self.ax2.imshow(drawland,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        plt.show()

    def initpic(self):
        initarr=np.ones([200,200])
        line1=self.ax1.imshow(initarr,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        line2=self.ax2.imshow(initarr,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        return line1, line2
   
    def calculate(self,index):
      
        a=self.lines[self.lineindex].split(' ',1)
        strmeasure=a[1].split()
        fltmeasure=list(map(float,strmeasure))
        measure=np.asarray(fltmeasure)
        self.lineindex=self.lineindex+10
        print (a[0])

        de=(fltmeasure-self.oldfltempty)*self.k2
        fltmeasure=de/self.k+self.fltempty

        d=np.divide(np.subtract(fltmeasure,self.fltempty),self.fltdelt)  #电容归一化

        ydn=self.dn.caul(d).T.reshape(-1)
        yland=self.land.caul(d).reshape(-1)
        t='tri'
        drawdn=self.draw.drawdata(ydn,t=t)
        drawland=self.draw.drawdata(yland,t=t)

        line1=self.ax1.imshow(drawdn,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        line2=self.ax2.imshow(drawland,cmap=plt.get_cmap('jet'),vmin=0,vmax=1)
        #line1=self.ax1.imshow(drawdn)
        #line1=self.ax1.imshow(drawland)

        return line1, line2


class DN():
    def __init__(self): 
        model = load_model('E:\\deeplearning\\程序\\ECT\\ECT\\logs\\DN190909-01\\weights.hdf5') 
        model.summary()
        self.model=model

    def caul(self,measure):
        self.measure=np.zeros([8,8])
        k=0
        for i in range(7):
            for j in range(i+1,8):
                self.measure[i][j]=measure[0][k]
                self.measure[j][i]=self.measure[i][j]
                k=k+1
        self.measure.shape=(1,8,8,1)
        y=self.model.predict(self.measure)
        y[y>1]=1
        y[y<0]=0
        return y

class Land():
    def __init__(self):
        self.path="E:\deeplearning\程序\ECT\ECT\真实实验";
        empty1=scipy.io.loadmat(self.path+'\empty.mat')
        efull1=scipy.io.loadmat(self.path+'\efull.mat')
        lmc1=scipy.io.loadmat(self.path+'\lmc.mat')
        self.fltempty=np.asarray(empty1['Cap'])    #空管标定数据
        self.fltfull=np.asarray(efull1['Cap'])    #满管标定数据
        self.lmc  =np.asarray(lmc1['S'])       #灵敏场
        self.Srow=np.sum(self.lmc,axis=0)#702
        self.Scol=np.sum(self.lmc,axis=1)#28
        self.capsize=28
        self.imgsize=834
        self.SLBP=np.zeros([self.capsize,self.imgsize])
        self.SLAND=np.zeros([self.capsize,self.imgsize])

        for i in range(self.imgsize):
            for j in range(self.capsize):
                self.SLBP[j][i]=self.lmc[j][i]/self.Srow[i]
        self.Srow2=np.sum(self.SLBP,axis=1)
        for i in range (self.capsize):
            for j in range(self.imgsize):
                self.SLAND[i][j]=self.SLBP[i][j]/self.Srow2[i]
        self.yLBP=np.zeros([1,self.imgsize])
        self.yLAND=np.zeros([1,self.imgsize])
        X= np.matrix(np.matmul(self.SLAND.T,self.SLAND))
        e, v = np.linalg.eig(X)
        ee=np.amax(e)
        self.a=(2/ee).real


    def caul(self,measure):
        g=np.matmul(self.SLAND.T,measure.T).reshape(-1,1)
        for k in range(100):
            m1=np.dot(self.SLAND,g)
            m2=m1-measure.T
            m3=np.dot(self.SLAND.T,m2)
            gk=g-self.a*m3
            gk[gk>1]=1
            gk[gk<0]=0
            g=gk
        return g
    
    
       
if __name__=='__main__':
    real=realplay()
    real.play()





