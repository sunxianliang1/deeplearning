path='E:\deeplearning\ECT\数据生成\data'
import glob
paths=glob.glob(path+'\data*.mat')
import scipy.io
data=scipy.io.loadmat(paths[0])


