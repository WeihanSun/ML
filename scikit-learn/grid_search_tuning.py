
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

parameters = { # set
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words' : ('english', None),
    'vect__max_features' : (2500, 5000, 10000, None),
    'vect__ngram_range' : ((1, 1), (1, 2)),
    'vect__use_idf' : (True, False),
    'vect__norm' : ('l1', 'l2'),
    'clf__penalty' : ('l1', 'l2'),
    'clf__C' : (0.01, 0.1, 1, 10)
}

if __name__ == '__main__':
    df = pd.read_csv("./data/ham_spam/SMSSpamCollection", delimiter='\t', header=None)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(df[1], df[0], test_size=0.25)
#    vectorizer = TfidfVectorizer()
#    X_train = vectorizer.fit_transform(X_train_raw)
#    X_test = vectorizer.transform(X_test_raw)
    grid_search.fit(X_train_raw, Y_train)
    print('Best score: %.3f' % grid_search.best_score_)
    print('Best parameters sets:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test_raw)