# Stochastic Gradient Descent

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_boston()
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target)

X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
Y_train = Y_scaler.fit_transform(Y_train.reshape(-1, 1))
X_test = X_scaler.transform(X_test)
Y_test = Y_scaler.transform(Y_test.reshape(-1, 1))

regressor_sgd = SGDRegressor(loss='squared_loss')
regressor_sgd.fit(X_train, Y_train)
#scores_sgd = cross_val_score(regressor_sgd, X_train, Y_train, cv=5)
#print('mean score = %.2f' % scores_sgd)