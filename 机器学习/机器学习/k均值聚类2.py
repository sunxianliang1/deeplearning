import numpy as np
from matplotlib import pyplot as plt 


x1=np.random.normal(1,0.4,100)
x2=x1+2
x=np.concatenate((x1,x2))

y1=np.random.normal(1,0.4,100)
y=np.zeros(200)
y[0:49]=y1[0:49]
y[100:149]=y1[0:49]
y[50:99]=y1[50:99]+2
y[150:199]=y1[50:99]+2



                  
plt.scatter(x,y,c = 'r')

plt.show()
plt.show()