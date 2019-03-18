import numpy as np
from random import shuffle
x = np.arange(60)
x.shape=(6,10)

y= np.arange(6)
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)
print(x)
print(y.T)

