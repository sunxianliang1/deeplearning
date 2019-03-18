import numpy as np
import matplotlib.pyplot as plt
import copy
def randkinds (kindnum,pointnum):
    "生成kindnum个二维点聚类，并展示图片,kindnum>2"
    "x y 范围均为0-10"
    dis=15/kindnum
    center=np.zeros([kindnum,2])
    for i in range(kindnum): 
        while True:
            point=np.random.rand(1,2)*10;
            j=0
            k=0
            while j<i:
                if (center[j][0]-point[0][0])**2+(center[j][1]-point[0][1])**2<dis**2:
                    k=1
                    break
                j=j+1       
            if k==0:
                center[i]=point[0]
                break
    points=np.zeros([kindnum,pointnum,2])
    points_nokind=np.zeros([kindnum*pointnum,2])
    k=0
    dis=dis/1.8
    for i in range (kindnum):
        for j in range (pointnum):
            while True:
                point=np.random.rand(1,2)*10;
                if (center[i][0]-point[0][0])**2+(center[i][1]-point[0][1])**2<dis**2:
                    points[i][j]=point[0]
                    points_nokind[k]=point[0]
                    k=k+1
                    break;
    plt.figure()
    plt.title("Init data")
    for i in range(kindnum):
        x=points[i,...,0]
        y=points[i,...,1] 
        col=(i+0.5)/kindnum 
        plt.scatter(x,y,c=(1,col,1-col)) 
    #plt.show()       
    return points,points_nokind

def k_means(points_nokind,kindnum):
    s=points_nokind.shape[0]
    centerpoints=np.zeros([kindnum,2])
    newcenterpoints=np.zeros([kindnum,2])
    
    for i in range(2):
        minj=np.amin(points_nokind[...,i])
        maxj=np.amax(points_nokind[...,i])
        for j in range(kindnum):
            centerpoints[j][i]=minj+(maxj-minj)*np.random.rand()
    cc=0
    while True:
        cc=cc+1
        count=np.zeros([kindnum],dtype=np.dtype('i2'))
        point_withkind=np.zeros([kindnum,s,2])
        #初次分类
        for i in range(s):
            mindis=10000000
            mincount=-1
            for j in range(kindnum):
                dis=(points_nokind[i][0]-centerpoints[j][0])**2+(points_nokind[i][1]-centerpoints[j][1])**2
                if dis<mindis:
                    mindis=dis
                    mincount=j
            point_withkind[mincount][count[mincount]]=points_nokind[i]
            count[mincount]=count[mincount]+1
        #重算质心
        for i in range(kindnum):
            pp=point_withkind[i][0:count[i]]
            newcenterpoints[i][0]=np.mean(pp[...,0])         
            newcenterpoints[i][1]=np.mean(pp[...,1])
        if (newcenterpoints==centerpoints).all() or cc>10:
            print(cc)
            break
        centerpoints=copy.deepcopy(newcenterpoints)
    return point_withkind,count,centerpoints

if __name__ == '__main__':
    kindnum=3
    points,points_nokind=randkinds(kindnum,50)
    #x=points_nokind[...,0]
    #y=points_nokind[...,1]
    #plt.figure()
    #plt.scatter(x,y)

    points_withkind,count,center=k_means(points_nokind,3)
    plt.figure()
    plt.title("Class data")
    for i in range(kindnum):
        x=points_withkind[i,0:count[i],0]
        y=points_withkind[i,0:count[i],1] 
        col=(i+0.5)/kindnum 
        plt.scatter(x,y,c=(1,col,1-col))
    plt.scatter(center[...,0],center[...,1],c='black') 
    plt.show()

