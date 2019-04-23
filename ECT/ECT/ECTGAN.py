from loaddata   import ECTdata
import tensorflow 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input
import matplotlib.pyplot as mp
from tensorflow.keras.backend import set_session
class GAN:
    def __init__(self,t='fig'):      
        self.mydata=ECTdata('E:\deeplearning\ECT\数据生成\data')
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






        pass

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
    ect=GAN(t='tri')
    ect.train(500)
    i=0
    while i!=-1:
        i=input("输入图片序号")
        i=int(i)
        ect.printpic(i,t='tri')






