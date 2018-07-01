# Receiver Operation Curve drawing
# SPAM & HAM SMS classification


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('./data/ham_spam/SMSSpamCollection', delimiter='\t', header=None)
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(df[1], df[0], test_size=0.25)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, threshold = roc_curve(Y_test == 'spam', predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC=%.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Fall-out')
plt.ylabel('Recall')
plt.show()

