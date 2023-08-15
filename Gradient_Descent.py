#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig

# loading the data

data = np.loadtxt("A2Q2Data_train.csv", delimiter=',')
data = pd.DataFrame(data)

x = data.iloc[0:,0:100]
x = np.array(x)
x = np.transpose(x)
xt = np.transpose(x)

def linear_regression(x,y):
    temp = np.dot(x, xt)
    #temp = np.inv(temp)
    temp = np.linalg.inv(temp)
    w_hat = np.dot(np.dot(temp, x), y)
    return w_hat



y = np.array(data.iloc[0:,100:])


parameter = data.iloc[9999:, 0:100]
Data_type = object
parameter = np.array(parameter, dtype=Data_type)
parameter = np.transpose(parameter)
for i in range (0, 100):
    parameter[i][0] = 0
    

    
    
def cost(x,y,parameter):
    temp = np.transpose(parameter)
    cost = (1 / (2 * x.shape[0])) * (np.square(np.dot(temp,x) - np.transpose(y))).sum()
    return cost


iteration = 5000
learning_rate = 0.000001

a = np.zeros((iteration, 1))
b = np.zeros((iteration, 1))

w_hat = linear_regression(x, y)

for i in range (0, iteration):
    parameter = parameter - learning_rate*(2*(np.dot((np.dot(x,xt)),parameter)) - 2*(np.dot(x,y)))
    a[i] =  np.linalg.norm(parameter - w_hat)
    b[i] = i
    
    
plt.plot(b,a)
plt.xlabel('iteration')
plt.ylabel("w - w_hat")
plt.show()


