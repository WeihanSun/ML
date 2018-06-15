"""
====================================
Plotting Cross-Validated Predictions
====================================

This example shows how to use `cross_val_predict` to visualize prediction
errors.

"""
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.5)
lr.fit(X_train, Y_train)
score = lr.score(X_test, Y_test)
print("half training score = ", score)
lr.fit(boston.data, boston.target)
score = lr.score(boston.data, boston.target)
print("all training score = ", score)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
