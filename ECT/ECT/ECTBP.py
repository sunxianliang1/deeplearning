from loaddata   import ECTdata
import tensorflow 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input
import matplotlib.pyplot as mp

class BP:
    def __init__(self):      
        self.mydata=ECTdata('E:\deeplearning\ECT\数据生成\data')
        self.mydata.initsca()
        print("data init success!")

    def train(self,times):
        if 'session' in locals() and tensorflow.session is not None:
            print('Close interactive session')
            tensorflow.session.close()
        input=Input(shape=(self.mydata.capsize,))
        decoded = Dense(50, activation='relu')(input)
        decoded1 = Dense(100, activation='relu')(decoded)
        decoded2 = Dense(200, activation='relu')(decoded1)
        decoded3 = Dense(400, activation='relu')(decoded2)
        output=Dense(self.mydata.imgsize, activation='relu')(decoded3)

        model=Model(inputs=input,outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.mydata.captrain,self.mydata.imgtrain,epochs=times,shuffle=True,validation_data=(self.mydata.captest,self.mydata.imgtest))
        p=model.evaluate(self.mydata.captest,self.mydata.imgtest)
        print(p)

    def printpic(self,index):      
        self.mydata.drawsca(self.mydata.imgtest[index])
        y=model.predict(self.mydata.captest)
        self.mydata.drawsca(y[index])
        mp.show()

if __name__=='__main__':
    bp=BP()
    bp.train(100)
    bp.printpic(40)






