from loaddata   import ECTdata
import tensorflow 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input




mydata=ECTdata('E:\deeplearning\ECT\数据生成\data')

input=Input(shape=(mydata.imgsize,))
encoded = Dense(140, activation='relu')(input)
mid = Dense(mydata.capsize, activation='relu')(encoded)
decoded = Dense(140, activation='relu')(mid)
output=Dense(mydata.imgsize, activation='relu')(decoded)

model=Model(inputs=input,outputs=output)
model.compile(optimizer='adadelta', loss='mean_squared_error')
model.fit(mydata.imgtrain,mydata.imgtrain,epochs=100,shuffle=True,validation_data=(mydata.imgtest,mydata.imgtest))
p=model.evaluate(mydata.imgtest,mydata.imgtest)
print(p)











