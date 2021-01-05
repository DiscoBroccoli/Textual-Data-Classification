
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import imdb
import evaluation
import pickle

# get the train-test data
X_train, y_train, X_test, y_test = imdb.fetch_data()

# Get best hyper parameters by grid search

# Logistic regression
parameter = {
    'penalty': ['l2', 'l1'],
    'tol': [0.01, 0.001, 0.0001],
    'C': [0.01, 0.1, 1]
}

gscv = GridSearchCV(LogisticRegression(), parameter, n_jobs=-1, cv=5)
gscv.fit(X_train, y_train)

best_para = gscv.best_estimator_.get_params()
for param_name in sorted(parameter.keys()):
    print('%s %r' % (param_name, best_para[param_name]))

print("Validation Acc:", gscv.best_score_)

pickle.dump(gscv.best_estimator_, open("lr_imdb.model", 'wb'))


