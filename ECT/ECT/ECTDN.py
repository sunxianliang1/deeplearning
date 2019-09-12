import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as mp
import sys
import shutil
import time
import os
from datetime import datetime
from loaddata   import ECTdata
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.datasets import mnist
import tensorflow
from tensorflow.keras.backend import set_session
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping

class DN():
    def __init__(self,t='fig',sample=0):      
        self.mydata=ECTdata('E:\deeplearning\ECT\数据生成\data2',size=sample)
        self.mydata.initsca(t=t)
        print("data init success!")
        #关闭上次未完全关闭的会话
        if 'session' in locals() and tensorflow.session is not None:
            print('Close interactive session')
            tensorflow.session.close()
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True  #允许显存增长
        set_session(tensorflow.Session(config=config))
        print('GPU memory is allowed to growth.')
        tensorflow.keras.backend.clear_session()

    def buildmodel(self):
        model=Sequential()

        #model.add(MaxPooling2D((1,1))
        model.add(UpSampling2D((2,2),input_shape=(8,8,1)))
        model.add(Conv2D(8, 4, 1,activation='relu',padding='same'))
        model.add(Dropout(0.2))
        #model.add(LeakyReLU())
        #model.add(MaxPooling2D((1,1)))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(8, 4, 1,activation='relu',padding='same'))
        model.add(Dropout(0.2))
        #model.add(LeakyReLU())
        #model.add(MaxPooling2D((1,1)))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(4, 4, 1,activation='relu',padding='same'))
        model.add(Dropout(0.2))
        #model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2048,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.mydata.imgsize,activation='relu'))

        model.summary()
        self.model=model

    def train(self,times):
        self.buildmodel()
        self.model.compile(optimizer='Adam', loss='mse')

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        os.mkdir('./logs/'+TIMESTAMP)
        # log_dir：log_dir：保存日志文件的地址
        # histogram_freq：计算各个层激活值直方图的频率（每多少个epoch计算一次），如果设置为0则不计算。
        # tensorboard=TensorBoard(log_dir='./logs/'+TIMESTAMP,histogram_freq=1) 
   
        # filename：字符串，保存模型的路径
        # monitor：需要监视的值
        # verbose：信息展示模式，0或1
        # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
        #checkpointer = ModelCheckpoint(filepath="./logs/"+TIMESTAMP+"/weights.hdf5", verbose=1, save_best_only=True)
        
        # monitor：需要监视的量
        # patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
        # verbose：信息展示模式
        # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
        earlystop=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        
        shutil.copy('./ECTDN.py','./logs/'+TIMESTAMP+'/ECTDN.py')
                                    
        self.model.fit(self.trainX,self.mydata.imgtrain,epochs=times,shuffle=True
                       ,validation_data=(self.testX,self.mydata.imgtest)
                       ,callbacks =[earlystop])
        self.model.save("./logs/"+TIMESTAMP+"/weights.hdf5")
        print("Model has been saved.")

        start = time.clock()
        e=self.model.evaluate(self.testX,self.mydata.imgtest)
        end = time.clock()
        print("测试集损失为%f"%e)
        print("重建使用时间为：%f"%(end-start))



    def printpic(self,index,t='fig'):      
        self.mydata.drawsca(self.mydata.imgtest[index],t=t)
        y=self.model.predict(self.testX)
        y[y>1]=1
        y[y<0]=0
        self.mydata.drawsca(y[index],t=t)
        mp.show()
        s=np.std(y[index]-self.mydata.imgtest[index])
        print("标准差为%f"%s)

    def printLastTest(self,index,t='fig'):

        self.mydata.drawsca(self.mydata.lastTestimages[index],t=t)
        y=self.model.predict(self.lastTestX)
        y[y>0.5]=y[y>0.5]*2
        y[y>1]=1
        y[y<0]=0
        self.mydata.drawsca(y[index],t=t)
        mp.show()
        s=np.std(y[index]-self.mydata.lastTestimages[index])
        print("标准差为%f"%s)

        np.savetxt("data\lastTestimages.txt",self.mydata.lastTestimages,fmt='%f', delimiter=',',newline='\n')
        np.savetxt("data\predictimgs.txt",y,fmt='%f', delimiter=',',newline='\n')


    def toMatrix(self):
        trainlen=self.mydata.captrain.shape[0]
        self.trainX=np.zeros([trainlen,8,8])
        for n in range(trainlen):
            k=0
            for i in range(7):
                for j in range(i+1,8):
                    self.trainX[n][i][j]=self.mydata.captrain[n][k]
                    self.trainX[n][j][i]=self.trainX[n][i][j]
                    k=k+1
        testlen=self.mydata.captest.shape[0]
        self.testX=np.zeros([testlen,8,8])
        for n in range(testlen):
            k=0
            for i in range(7):
                for j in range(i+1,8):
                    self.testX[n][i][j]=self.mydata.captest[n][k]
                    self.testX[n][j][i]=self.testX[n][i][j]
                    k=k+1
        self.trainX.shape=(-1,8,8,1)
        self.testX.shape=(-1,8,8,1)

        len=4
        self.lastTestX=np.zeros([len,8,8])
        for n in range(len):
            k=0
            for i in range(7):
                for j in range(i+1,8):
                    self.lastTestX[n][i][j]=self.mydata.lastTestcaps[n][k]
                    self.lastTestX[n][j][i]=self.lastTestX[n][i][j]
                    k=k+1
        self.lastTestX.shape=(-1,8,8,1)
        


    def loadmodel(self):
        model = load_model('E:\\deeplearning\\程序\\ECT\\ECT\\logs\\DN190909-01\\weights.hdf5') 
        model.summary()
        self.model=model


if __name__=='__main__':
    bp=DN(t='tri')
    bp.toMatrix()
    #bp.train(100)
    bp.loadmodel()

    while True:
        i=input("输入图片序号")
        i=int(i)
        if (i>=0):
            bp.printLastTest(i,t='tri')                                     
        else:
            break



