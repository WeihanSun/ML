# SMS spam classification

import pandas as pd
df = pd.read_csv('./data/ham_spam/SMSSpamCollection', delimiter='\t', header=None)
print(df.head())
print('Num. of spam message:', df[df[0] == 'spam'][0].count())
print('Num. of ham message', df[df[0] == 'ham'][0].count())

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(df[1], df[0], test_size=0.2)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)
for i, prediction in enumerate(predictions[:5]):
    print('Prediction: %s. %s' % (prediction, np.array(X_test_raw)[i]))

# evaluation
# confusion matrix
# accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test, predictions)
print(cm)
accuracy = accuracy_score(Y_test, predictions)
print('Accuracy = %.2f%%' % (accuracy * 100))
plt.matshow(cm)
plt.title('confusion_matrix')
plt.colorbar()
plt.text(0,0, 'TN', style='italic', bbox={'facecolor':'red', 'alpha':1, 'pad':10})
plt.text(1,0, 'FP', style='italic', bbox={'facecolor':'red', 'alpha':1, 'pad':10})
plt.text(0,1, 'FN', style='italic', bbox={'facecolor':'red', 'alpha':1, 'pad':10})
plt.text(1,1, 'TP', style='italic', bbox={'facecolor':'red', 'alpha':1, 'pad':10})

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# cross validation (precision & recall)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, Y_train, cv=5)
print('cross validation : ', scores)
#precision = cross_val_score(classifier, X_train, Y_train, cv=5, scoring='precision')
#print('precision : ', precision)
#recall = cross_val_score(classifier, X_train, Y_train, cv=5, scoring='recall')
#print('recall : ', recall)

