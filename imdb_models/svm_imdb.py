
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import pickle
import imdb

# get the train-test data
X_train, y_train, X_test, y_test = imdb.fetch_data(reset=True)

# Get best hyper parameters by grid search

parameter = {
    'loss': ['hinge', 'squared_hinge'],
    'tol': [0.01, 0.001, 0.0001],
    'C': [0.01, 0.1, 1]
}

gscv = GridSearchCV(LinearSVC(), parameter, n_jobs=-1, verbose=10, cv=5)
gscv.fit(X_train, y_train)

best_para = gscv.best_estimator_.get_params()
for param_name in sorted(parameter.keys()):
    print('%s %r' % (param_name, best_para[param_name]))
print("Validation Acc:", gscv.best_score_)

pickle.dump(gscv.best_estimator_, open("svm_imdb.model", 'wb'))
