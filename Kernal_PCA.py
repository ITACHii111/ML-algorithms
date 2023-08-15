import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
import math as mt
import sys

data=pd.read_csv("Dataset.csv",header=None)
data_set=np.array(data)


a=mt.exp(1)
covar=np.random.randint(1,size=(1000,1000))
x=range(1000)
sigma=0.1
temp2=0
temp3=0
temp4=0
temp5=0
covari = np.arange(1000000).reshape(1000,1000)
covari = np.float64(covari)
for m in range(10):
    for i in range(1000):
        for j in range(1000):
            temp=data_set[i]-data_set[j]
            temp1=temp
            temp=np.transpose(temp1)
            temp2=-(temp.dot(temp1))
            temp3=2*(((m+1)*sigma)**2)
            temp4=temp2/temp3
            temp5=a**temp4
            covari[i][j]=np.float64(temp5)
    temp1=np.random.randint(1, size=(1000,1000))
    temp1=temp1+1/1000
    temp2=temp1.dot(covari)
    temp3=covari.dot(temp1)
    temp4=temp2.dot(temp1)
    covari=covari-temp2-temp3+temp4
    eig_val,eig_vec=eig(covari)
    duplicate_val=eig_val
    duplicate_val.sort()
    a=duplicate_val[999]
    b=duplicate_val[998]
    y=range(1000)
    vector1=eig_vec[0]
    vector2=eig_vec[1]
    for i in y:
        if(duplicate_val[i]==a):
            vector1=eig_vec[i]
        if(duplicate_val[i]==b):
            vector2=eig_vec[i]
    vector1=vector1/mt.sqrt(a)
    vector2=vector1/mt.sqrt(b)
    trans=np.transpose(covari)
    proj_1=trans.dot(vector1)
    proj_2=trans.dot(vector2)
    plt.scatter(proj_1,proj_2)
    plt.title("projection of dataset")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
