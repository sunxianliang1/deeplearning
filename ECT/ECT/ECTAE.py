from loaddata   import ECTdata
import tensorflow 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input
import matplotlib.pyplot as mp

#双向训练
mydata=ECTdata('E:\deeplearning\ECT\数据生成\data',10000)
mydata.initsca()
print("data init success!")

input=Input(shape=(mydata.imgsize,))
encoded = Dense(140, activation='relu')(input)
midnet=Dense(mydata.capsize, activation='relu')
mid = midnet(encoded)
midoutput=midnet.output
decoded = Dense(140, activation='relu')(mid)
output=Dense(mydata.imgsize, activation='relu')(decoded)


model=Model(inputs=input,outputs=[midoutput,output])
model1=Model(inputs=input,outputs=output)

model.compile(optimizer='adadelta', loss='mean_squared_error')
model.fit(mydata.imgtrain,[mydata.captrain,mydata.imgtrain],epochs=500,shuffle=True)
p=model.evaluate(mydata.imgtest,[mydata.captest,mydata.imgtest])
print("整个网络的损失为%f %f"%(p[0],p[1]))

mid2=Input(shape=(mydata.capsize,))
decoded1=model.layers[-2](mid2)
decoded2=model.layers[-1](decoded1)
model2=Model(inputs=mid2,outputs=decoded2)
model2.compile(optimizer='adadelta', loss='mean_squared_error')
p2=model2.evaluate(mydata.captest,mydata.imgtest)
print("后半个网络的损失为%f"%p2)


index=1
mydata.drawsca(mydata.imgtest[index])
y1=model.predict(mydata.imgtest)
y2=model2.predict(mydata.captest)
mydata.drawsca(y1[1][index])
mydata.drawsca(y2[index])
mp.show()






