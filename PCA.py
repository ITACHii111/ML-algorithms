import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

ap=pd.read_csv("Dataset.csv",header=None) # taking the input in matrix form
data_set=np.array(ap)

feature1=ap[0].mean()  # mean of dimension 1
feature2=ap[1].mean()  # mean of dimension 2

x=range(1000)                      # data centralising                   
for i in x:
    data_set[i][0]=data_set[i][0]-feature1
    data_set[i][1]=data_set[i][1]-feature1

for i in range(1000):
    plt.scatter(data_set[i][0],data_set[i][1],c="blue")
plt.title("Pictorial Representation of given Data")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

duplicate_data=np.transpose(data_set)
covar=duplicate_data.dot(data_set)
covar=covar/1000

eig_value,eig_vector=eig(covar)
PCA0=eig_vector[0]
PCA1=eig_vector[1]

m=PCA1[1]/PCA1[0]
x=np.array([-5,5])
y=m*x
k=PCA0[1]/PCA0[0]
x1=np.array([-5,5])
y1=k*x1

for i in range(1000):
    plt.scatter(data_set[i][0],data_set[i][1],c="blue")
plt.title("Component of PCA along dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x,y)
plt.plot(x1,y1)

m=(data_set)
m1=m.dot(PCA1)
m2=m.dot(PCA0)
x=range(1000)
var=0
for i in x:
    var=var+(m1[i])**2
var=var/1000
var1=0
x=range(1000)
for i in x:
    var1=var1+(m2[i])**2

var1=var1/1000
print("variance along component 1 is", var)
print("variance along component 2 is", var1)