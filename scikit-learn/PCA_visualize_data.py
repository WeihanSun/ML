# PCA to reduce the dimension of the data
# iris data set : 3 kinds, 4-dim features
# reduce to 2-dim features and plot to Cartesian coordination,
# colored corresponding with their categories


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data = load_iris()
Y = data['target']
X = data['data']
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    if Y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif Y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
s1 = plt.scatter(red_x, red_y, c='r', marker='o')
s2 = plt.scatter(blue_x, blue_y, c='b', marker='D')
s3 = plt.scatter(green_x, green_y, c='g', marker='^')
plt.legend('lower right')
plt.legend((s1, s2, s3), ('iris 0', 'iris 1', 'iris 2'))
plt.show()


