from loaddata   import ECTdata
import tensorflow 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input
import matplotlib.pyplot as mp
from tensorflow.keras.backend import set_session
class BP:
    def __init__(self,t='fig'):      
        self.mydata=ECTdata('E:\deeplearning\ECT\数据生成\data',5000)
        self.mydata.initsca(t=t)
        print("data init success!")

    def train(self,times):
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True  #允许显存增长
        set_session(tensorflow.Session(config=config))
        if 'session' in locals() and tensorflow.session is not None:
            print('Close interactive session')
            tensorflow.session.close()

        input=Input(shape=(self.mydata.capsize,))
        decoded = Dense(256, activation='relu')(input)
        decoded1 = Dense(512, activation='relu')(decoded)
        #decoded2 = Dense(200, activation='relu')(decoded1)
        #decoded3 = Dense(400, activation='relu')(decoded2)
        #output=Dense(self.mydata.imgsize, activation='relu')(decoded3)
        decoded2 = Dense(1024, activation='relu')(decoded1)
        output=Dense(self.mydata.imgsize, activation='tanh')(decoded2)


        self.model=Model(inputs=input,outputs=output)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.mydata.captrain,self.mydata.imgtrain,epochs=times,shuffle=True,validation_data=(self.mydata.captest,self.mydata.imgtest))
        p=self.model.evaluate(self.mydata.captest,self.mydata.imgtest)
        self.model.save("BP4layer.h5")
        print(p)

    def printpic(self,index,t='fig'):      
        self.mydata.drawsca(self.mydata.imgtest[index],t=t)
        y=self.model.predict(self.mydata.captest)
        y[y>1]=1
        y[y<0]=0
        self.mydata.drawsca(y[index],t=t)
        mp.show()

if __name__=='__main__':
    bp=BP(t='tri')
    bp.train(50)
    i=0
    while i!=-1:
        i=input("输入图片序号")
        i=int(i)
        bp.printpic(i,t='tri')






