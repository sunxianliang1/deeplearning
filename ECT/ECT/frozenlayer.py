from loaddata   import ECTdata
import tensorflow 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input
import matplotlib.pyplot as mp
from tensorflow.keras.backend import set_session
import numpy as np
#关闭上次未完全关闭的会话
if 'session' in locals() and tensorflow.session is not None:
    print('Close interactive session')
    tensorflow.session.close()
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True  #允许显存增长
set_session(tensorflow.Session(config=config))
print('GPU memory is allowed to growth.')
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = True
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = False
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

data=np.random.normal(0, 1, (100, 32))
labels=np.random.normal(0, 1, (100, 32))
frozen_model.fit(data, labels,epochs=5)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels,epochs=5)  # this updates the weights of `layer`
frozen_model.fit(data, labels,epochs=5)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels,epochs=5)  # this updates the weights of `layer`
frozen_model.fit(data, labels,epochs=5)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels,epochs=5)  # this updates the weights of `layer`