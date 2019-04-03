from loaddata   import ECTdata
import matplotlib.pyplot as mp
import numpy as np

mydata=ECTdata('E:\deeplearning\ECT\数据生成\data')
mydata.initsca()
a=mydata.index.index(2)
mydata.drawsca(mydata.images[a])
print(mydata.images[a].sum())
mp.show()



