import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as mp
import sys
from loaddata   import ECTdata
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
import tensorflow
from tensorflow.keras.backend import set_session


class DN():
    def __init__(self,t='fig'):      
        self.mydata=ECTdata('E:\deeplearning\ECT\数据生成\data',5000)
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
    def model(self):
        model=Sequential()
        model.add()




    def train(self,times):


        pass
    def printpic(self,index,t='fig'):      
        self.mydata.drawsca(self.mydata.imgtest[index],t=t)
        y=self.model.predict(self.mydata.captest)
        y[y>1]=1
        y[y<0]=0
        self.mydata.drawsca(y[index],t=t)
        mp.show()

if __name__=='__main__':
    bp=DN(t='tri')
    bp.train(500)
    i=0
    while i!=-1:
        i=input("输入图片序号")
        i=int(i)
        bp.printpic(i,t='tri')




