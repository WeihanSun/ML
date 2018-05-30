# 2 parameters' linear regression
# X: pizza size & topping num.; Y: pizza price

from numpy.linalg import inv
from numpy import dot, transpose
import numpy as np
from sklearn.linear_model import LinearRegression

# 1 is to get intercept
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
Y = [[7], [9], [13], [17.5], [18]]

# linear equation
#print(dot(inv(dot(transpose(X), X)), dot(transpose(X), Y)))
# numpy lib func
#print(np.linalg.lstsq(X, Y, rcond=None)[0])
# sklearn
X1 = np.array(X)[:,1:3].tolist()
model = LinearRegression()
model.fit(X1, Y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
Y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)
print(predictions)
for i, prediction in enumerate(predictions):
    print('Predicted: %s, Target: %s' % (prediction, Y_test[i]))
print('R-squared(train): %.2f' % model.score(X1, Y))
print('R-squared(test): %.2f' % model.score(X_test, Y_test))


