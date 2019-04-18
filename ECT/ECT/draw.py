from loaddata   import ECTdata
import matplotlib.pyplot as mp
import numpy as np

mydata=ECTdata('E:\deeplearning\ECT\数据生成\datatest')
mydata.initsca(t='tri')
for i in  range(2):
    mydata.drawsca(mydata.images[i],t='tri')
mp.show()



