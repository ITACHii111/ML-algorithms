import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

data = pd.read_csv("A2Q2Data_train.csv");
data = np.array(data)


v = np.zeros((10000, 100))
for x in range(0,9999):
    for y in range(0,99):
        v[x][y] = data[x][y]
v = np.transpose(v)
trans = np.transpose(v)
result = np.dot(v,trans)
inv = np.linalg.pinv(result)
prediction = np.zeros((10000,1))
for i in range(0,9999):
    prediction[i][0] = data[i][100]
ans = np.dot(inv, v)
ans = np.dot(ans,prediction)