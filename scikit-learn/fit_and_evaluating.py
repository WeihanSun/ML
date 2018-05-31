# apply linear regression to red wine quality evaluation
# cross validation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

data = pd.read_csv("data/winequality-red.csv", sep=';')
print(data.describe())

plt.title('Alcohol against Quality')
plt.scatter(data['alcohol'], data['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.show()

X = data[list(data.columns)[:-1]]
Y = data['quality']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print('r squared value(1 test) = %.2f' % regressor.score(X_test, Y_test))

regressor_cv = LinearRegression()
scores = cross_val_score(regressor_cv, X, Y, cv=5)
print('mean score = %.2f' % scores.mean())
np.set_printoptions(formatter={'float': '{:.2f}'.format})
print('cross validation score = ', scores)
