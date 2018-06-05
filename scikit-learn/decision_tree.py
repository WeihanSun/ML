import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    df = pd.read_csv('./data/ad/ad.data', header=None, low_memory=False)
    explanatory_variable_columns = set(df.columns.values)
    response_variable_column = df[len(df.columns.values) - 1]
    explanatory_variable_columns.remove(len(df.columns.values)-1)
    Y = [1 if e == 'ad.' else 0 for e in response_variable_column]
    X = df[list(explanatory_variable_columns)]
    # missing data
    X.replace(to_replace=' ', value=-1, regex=True, inplace=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    pipeline = Pipeline(['cls', DecisionTreeClassifier(criterion='entropy')])
    parameters = {
        'clf__max_depth': (150, 155, 160),
        'clf__min_smaples_split': (1, 2, 3),
        'clf__min_samples_leaf' : (1, 2, 3)
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train, Y_train)
    print('Best score: %.3' % grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    predictions = grid_search.predict(X_test)
    print(classification_report(Y_test, predictions))