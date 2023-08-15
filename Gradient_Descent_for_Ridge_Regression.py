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




def linear_regression(x, xt, label):
    temp = np.dot(x, xt)
    #temp = np.inv(temp)
    temp = np.linalg.inv(temp)
    w = np.dot(np.dot(temp, x), label)
    return w





parameter = np.zeros((100,1))
learning_rate = 0.000001
iteration = 200





def cost(x,y,parameter,lamda):
    xt = x.transpose()
    xtw = np.matmul(xt,parameter)
    yp = xtw - y
    yp = np.dot(yp.transpose(), yp)
    r = lamda*(np.square(np.linalg.norm(parameter)))
    yp = yp/(2*x.shape[0])
    tcost = yp + r
    return tcost
    

def ridge_regression(x,y,parameter,learning_rate,iteration,lamda):
    temp = np.transpose(x)  
    for i in range (0, iteration):
        parameter = parameter - 2*learning_rate*(np.dot(np.dot(x,temp),parameter) - (np.dot(x,y))+ (lamda*parameter))
    reduced_cost = cost(x,y,parameter,lamda)
    return reduced_cost,parameter;
    

batch = 10
a = np.zeros((batch-1, 1))
b = np.zeros((batch-1, 1))
w_ridge = np.zeros((100,1))
temp = np.zeros((100,1))


for i in range (1, batch):
    lamda = i*0.1
    tot_cost,temp= ridge_regression(x,label,parameter,learning_rate,iteration,lamda)
#     print(tot_cost)
    if(i == 1):
        w_ridge = temp
    a[i-1] = tot_cost
    b[i-1] = lamda
    
    
plt.plot(b, a)
plt.xlabel("Different λ values")
plt.ylabel("Error function according to λ")
plt.show()


w_hat = linear_regression(x,xt,label)


# best lamda with the least error possible is 0.1


file = pd.read_csv("A2Q2Data_test.csv", delimiter=',')
# print(file)
file = np.array(file)
# print(file.shape)
x_test = file[0:,0:100]
x_test = np.array(x_test)
x_test = np.transpose(x_test)
y_test = label = np.array(file[0:,100:])
w = np.zeros((100,1))
total_cost1 = cost(x_test,y_test,w_ridge,0)
total_cost2 = cost(x_test,y_test,w_hat,0)
print("Error for ridge regression ",total_cost1)                                         # error using ridge regression
print("Error for the analytical solution ",total_cost2)                                  # error using analytical solution (linear regression)
