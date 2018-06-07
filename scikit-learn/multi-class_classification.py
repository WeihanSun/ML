# one-vs-all to classify multi-class
# movie review with sentiments as negative(0), somewhat negative(1), neutral(2),
# somewhat positive(3), positive(4)

import pandas as pd
df = pd.read_csv('./data/mov-review/train.tsv', header=0, delimiter='\t')
print(df.count())
print(df.head(3))
print(df['Sentiment'].head(10))
print(df['Sentiment'].describe())
print(df['Sentiment'].value_counts()/df['Sentiment'].count())
print("=========================")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.25, 0.5),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'clf__C': (0.1, 1, 10)
}
X, Y = df['Phrase'], np.array(df['Sentiment'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
grid_search.fit(X_train, Y_train)
print('Best score: %.3f' % grid_search.best_score_)
print('Best parameters set: ')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))


