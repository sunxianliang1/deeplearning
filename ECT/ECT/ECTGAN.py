from loaddata   import ECTdata
import tensorflow 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import matplotlib.pyplot as mp
from tensorflow.keras.backend import set_session
import numpy as np
import warnings
import os

class GAN:
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
        '''建立完整模型'''
        optimizer = Adam(0.0002, 0.5)
        #生成器generator
        inputGen=Input(shape=(self.mydata.capsize,))
        g=Sequential()
        g.add(Dense(256,activation='sigmoid'))     
        g.add(Dense(512,activation='sigmoid'))
        g.add(Dense(1024,activation='sigmoid'))
        g.add(Dense(self.mydata.imgsize,activation='tanh'))
        g1=g(inputGen)
        self.Generator=Model(inputs=inputGen,outputs=g1)
        #self.Generator.compile(loss='binary_crossentropy',optimizer='adam')

        #判别器discriminator
        inputDis=Input(shape=(self.mydata.imgsize,))
        d=Sequential()
        d.add(Dense(512,activation='sigmoid')) 
        d.add(Dense(256,activation='sigmoid'))             
        d.add(Dense(1,activation='sigmoid'))
        d1=d(inputDis)
        self.Discriminator=Model(inputs=inputDis,outputs=d1)
        self.Discriminator.trainable=True
        self.Discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        #总体
        Generateimg=self.Generator(inputGen)
        self.Discriminator.trainable=False
        validity = self.Discriminator(Generateimg)
        self.combined = Model(inputGen, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
 

    def train(self, epochs, batch_size=100, sample_interval=50):
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        batchs=len(self.mydata.imgtrain)//batch_size
        for epoch in range(epochs):

            for batch in range(batchs):
                '''  Train Discriminator   '''
                #Select a batch of data
                #realcap=self.mydata.captrain[batch*batch_size:(batch+1)*batch_size]
                realcap=np.random.uniform(0, 1, size=(batch_size, 28))
                realimg=self.mydata.imgtrain[batch*batch_size:(batch+1)*batch_size]
                # Generate a batch of new images
                gen_img = self.Generator.predict(realcap)
                #img=np.concatenate((realimg,gen_img),axis=0)
                #state = np.random.get_state()
                #np.random.shuffle(img)
                #label=np.append(valid,fake)
                #np.random.set_state(state)
                #np.random.shuffle(label)
                
                # Train the discriminator
                #d_loss = self.Discriminator.train_on_batch(img, label)
                d_loss_real = self.Discriminator.train_on_batch(realimg, valid)
                d_loss_fake = self.Discriminator.train_on_batch(gen_img, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                '''  Train Generator   '''
                realcap=np.random.uniform(0, 1, size=(batch_size, 28))
                g_loss = self.combined.train_on_batch(realcap, valid)
                
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    def printpic(self,index,t='fig'):      
        self.mydata.drawsca(self.mydata.imgtest[index],t=t)
        y=self.Generator.predict(self.mydata.captest[index:index+1])
        y[y>1]=1
        y[y<0]=0
        self.mydata.drawsca(y[0],t=t)
        mp.show()

if __name__=='__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
    ect=GAN(t='tri')
    ect.model()
    ect.train(500)
    while True:
        i=input("输入图片序号")
        i=int(i)
        if i<0:
            break
        ect.printpic(i,t='tri')






