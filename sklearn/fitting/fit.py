# coding: utf-8
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pickle
data = pickle.load(open('black.learndata','rb'))
data
x, y = zip(data.values())
x, y = zip(*data.values())
x
x = np.array(x)
x
y = np.array(y)
regr = DecisionTreeRegressor(max_depth=5)
regr.fit(x,y)
x.shape
x.reshape(1898, -1)
newx = x.reshape(1898, -1)
newx[0]
regr.fit(newx, y)
y1 = regr.predict(newx)
y1
y
plt.plot(y, y1)
plt.savefig('fitting.pdf')
