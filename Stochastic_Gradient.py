import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig


data = np.loadtxt("A2Q2Data_train.csv", delimiter=',')
data = pd.DataFrame(data)
data = np.array(data)
data.shape

x = data[0:,0:100]
x = np.array(x)
x = np.transpose(x)
xt = np.transpose(x)

label = np.array(data[0:,100:])



temp = np.random.randint(1, size=(100, 101))
w = np.zeros((100,1))


parameter = np.zeros((100,1))



learning_rate = 0.0001
iteration = 100


t = 300
c = np.zeros((t,1))
b = np.zeros((t,1))



def cost(x,y,parameter):
    temp = np.transpose(parameter)
    cost = (1 / (2 * x.shape[0])) * (np.square(np.dot(temp,x) - np.transpose(y))).sum()
    return cost



def linear_regression(x, xt, y):
    temp = np.dot(x, xt)
    #temp = np.inv(temp)
    temp = np.linalg.inv(temp)
    w_hat = np.dot(np.dot(temp, x), y)
    return w_hat


w_hat = linear_regression(x, xt, label)


def linear_regression_using_stochastic_gradient(x, label, temp, data, learning_rate, iteration, w, w_hat,t):
    #print(w.shape)
    for j in range (0, t):
        a = np.random.randint(9999, size = 100)
        y = np.zeros((100,1))
        idx = np.random.randint(9999, size=100)
        temp = data[idx,:]
        y = temp[:, -1] 
        y = y.reshape(100,1)
        temp = np.delete(temp,-1,1)
        parameter = np.zeros((100,1))
        for i in range (0, iteration):
            dldw = (2 * learning_rate * (np.matmul(np.matmul(temp, np.transpose(temp)),parameter) - np.matmul(temp,y)))
            parameter = parameter - dldw 
            #w = w + parameter
        
        
        c[j] = np.linalg.norm(parameter - w_hat)
        b[j] = j
    plt.plot(b,c)
    plt.xlabel('T')
    plt.ylabel("w - w_hat as a function of T")
    plt.show()
    
        
        

w_t = linear_regression_using_stochastic_gradient(x, label, temp, data, learning_rate, iteration, w, w_hat, t)


