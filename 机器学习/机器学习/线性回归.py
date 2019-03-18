
##https://blog.csdn.net/July_sun/article/details/53223962
import numpy as np 
from matplotlib import pyplot as plt 
 
x = np.arange(-10,11).T 
z=(np.random.rand(21)-0.5)*4
y =  0.2* x*x +  0.5*x + 3+z
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y,"ob") 


#最小二乘法  
  #求导法    J(c)=1/2m *sum(Ym-Y)^2      i=1:m数据量  j=1:n  x维度
  #          dJ(c)/dc=1/m *sum(Ym-Y)*Xj^i=0   ->
phi=np.zeros([21,3])
for i in range(3):
    phi[...,i]=x**i
A=np.zeros([3,3])
Ym=np.zeros([3,1])
for i in range(3):
    for j in range(3):
        A[i][j]=np.vdot(phi[...,i],phi[...,j])
    Ym[i]=np.vdot(y,phi[...,i])
Ainv=np.linalg.inv(A)
c=np.dot(Ainv,Ym)
print(c)
ym=c[0]+c[1]*x+c[2]*x*x
plt.plot(x,ym) 


#A'Ax=A'b
A1=np.dot(phi.T,phi)
b1=np.dot(phi.T,y)
c1=np.linalg.solve(A1,b1)
print(c1)


#梯度下降法   迭代
#      cj=cj- k* dJ(c)/dc
c2=np.zeros([3,1])
i=1
k=0.0005
while i<20000:
    h=np.dot(phi,c2)
    h=h.flat-y
    for j in range(3):     
        hj=np.vdot(h,phi[...,j])
        r=1/21*hj
        c2[j]=c2[j]*(1-k/21)-k*r
    i=i+1

print(c2)
ym2=c2[0]+c2[1]*x+c2[2]*x*x
plt.plot(x,ym2) 
plt.show()

##收敛极慢，且对k值要求较高，否则会发散
