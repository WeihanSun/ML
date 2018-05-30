# linear regression
# X: pizza size; Y: pizza price
# R-squared, ddof

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

X = [[6], [8], [10], [14], [18]]
Y = [[7], [9], [13], [17.5], [18]]
X_test = [[8], [9], [11], [16], [12]]
Y_test = [[11], [8.5], [15], [18], [11]]

# Create model
model = LinearRegression()
model.fit(X, Y)
#print("A 12 should be %.2f" % model.predict([12][0]))
print('Residual sum of square: %.2f' % np.mean((model.predict(X) - Y) ** 2))
print('variance = %.2f' % np.var(X, ddof=1))
print('mean (X) = %.2f' % np.mean(X))
print('mean (Y) = %.2f' % np.mean(Y))
print('R-squared(train) : %.4f' % model.score(X, Y))
print('R-squared(test) : %.4f' % model.score(X_test, Y_test))

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, Y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)

# regression line
Xt = [[i] for i in range(0, 25)]
plt.plot(Xt, model.predict(Xt), 'r-')
plt.show()