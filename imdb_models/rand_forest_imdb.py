
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import imdb
import evaluation
import numpy as np
import pickle

# get the train-test data
X_train, y_train, X_test, y_test = imdb.fetch_data()

# Get best hyper parameters by grid search

# Naive Bayes
parameter = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [100, 150, 200],
    'max_features': ['auto', 'sqrt'],
}

gscv = GridSearchCV(RandomForestClassifier(), parameter, n_jobs=-1, verbose=10, cv=5)
gscv.fit(X_train, y_train)

best_para = gscv.best_estimator_.get_params()
for param_name in sorted(parameter.keys()):
    print('%s %r' % (param_name, best_para[param_name]))

print("Validation Acc:", gscv.best_score_)

pickle.dump(gscv.best_estimator_, open("forest_imdb.model", 'wb'))

# Evaluate performance of each model on best parameters
print("Random Forests")
evaluation.evaluate_model(gscv.best_estimator_, X_train, y_train, X_test, y_test)



# Validation Curves
#evaluation.plot_validation_curve(gscv.best_estimator_, "title", X_train, y_train, "max_iter", np.linspace(0, 50, num=11))
#plt.show()
