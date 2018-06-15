# classify handwritten digits(0~9) using SVM
# MNIST: 70,000 images 28x28 pixels in grayscale

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

digits = fetch_mldata('MNIST original', data_home='data/mnist')
X = digits['data']
counter = 1
for i in range(1, 4):
    for j in range(1, 6):
        plt.subplot(3, 5, counter)
        plt.imshow(X[(i - 1) * 8000 + j].reshape((28, 28)), cmap=cm.Greys_r)
        plt.axis('off')
        counter += 1
plt.show()

X = X/255*2 - 1
Y = digits['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y);
pipline = Pipeline([('clf', SVC(kernel='rbf'))])
parameters = {
    'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
    'clf__C': (0.1, 0.3, 1, 3, 10, 30)
}

grid_search = GridSearchCV(pipline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train[:10000], Y_train[:10000])
print('Best score: %.3f' % grid_search.best_score_)
print('Best parameters :')
best_parameters = grid_search.best_estimator_.get_params()
#best_parameters = grid_search.best_params_
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print(classification_report(Y_test, predictions))
print(classification_report(predictions, Y_test))
