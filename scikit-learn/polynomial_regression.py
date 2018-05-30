# polynomial regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]

X_test = [[6], [8], [11], [16]]
Y_test = [[8], [12], [15], [18]]

xx = np.linspace(0, 26, 100)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=2)
# fit needed for first time; fit() must before transform()
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, Y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy_quadratic = regressor_quadratic.predict(xx_quadratic)

print('==train data==')
print(X_train)
print(X_train_quadratic)
print('==test data==')
print(X_test)
print(X_test_quadratic)

plt.plot(xx, yy_quadratic, c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, Y_train)
plt.show()

print("R-squared(1): %.2f" % regressor.score(X_test, Y_test))
print("R-sqqured(2): %.2f" % regressor_quadratic.score(X_test_quadratic, Y_test))