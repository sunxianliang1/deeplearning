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

class GAN():
    def __init__(self,t='fig'):
        self.mydata=ECTdata('E:\deeplearning\ECT\数据生成\data',5000)
        self.mydata.initsca(t=t)
        print("data init success!")

        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True  #允许显存增长
        set_session(tensorflow.Session(config=config))

        optimizer =Adam()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mean_squared_error',optimizer=optimizer)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.mydata.capsize,))
        img = self.generator(z)

        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.mydata.capsize))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.mydata.imgsize, activation='tanh'))      
        model.summary()#打印网络结构

        noise = Input(shape=(self.mydata.capsize,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(512,input_dim=self.mydata.imgsize))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='tanh'))
        model.summary()
        img = Input(shape=(self.mydata.imgsize,))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=100):


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        batchs=len(self.mydata.imgtrain)//batch_size
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for batch in range(batchs):
                #index=np.random.randint(0,4000,size=(batch_size,))
                noise=self.mydata.captrain[batch*batch_size:(batch+1)*batch_size]*2-1
                #noise=np.random.normal(0,1,size=(batch_size,28))有监督学习网络
                gen_imgs = self.generator.predict(noise)
                imgs=self.mydata.imgtrain[batch*batch_size:(batch+1)*batch_size]
                        
                # Train the discriminator
                #for _ in range (1):
                #    d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                #    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                #    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

            
                # For the combined model we will only train the generator
            
                # Train the generator (to have the discriminator label samples as valid)
                gloss=self.generator.train_on_batch(noise,imgs)
                #gloss=1
                #g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
           # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f,All loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], gloss,g_loss))
            print ("%d [G loss: %f]" % (epoch,  gloss))

    def printpic(self,index,t='fig'):      
        self.mydata.drawsca(self.mydata.imgtest[index],t=t)
        y=self.generator.predict(self.mydata.captest[index:index+1]*2-1)
        y[y>1]=1
        y[y<0]=0
        self.mydata.drawsca(y[0],t=t)
        mp.show()

    


if __name__ == '__main__':
    tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
    gan = GAN(t='tri')
    gan.train(epochs=50, batch_size=100)
    while True:
        i=input("输入图片序号")
        i=int(i)
        if i<0:
            break
        gan.printpic(i,t='tri')


