
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import imdb
import evaluation
import pickle
import matplotlib.pyplot as plt
import numpy as np

# get the train-test data
X_train, y_train, X_test, y_test = imdb.fetch_data()

# Get best hyper parameters by grid search

parameter = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.1, 1, 10]
}

gscv = GridSearchCV(AdaBoostClassifier(), parameter, n_jobs=-1, verbose=10, cv=5)
gscv.fit(X_train, y_train)

best_para = gscv.best_estimator_.get_params()
for param_name in sorted(parameter.keys()):
    print('%s %r' % (param_name, best_para[param_name]))

print("Validation Acc:", gscv.best_score_)


pickle.dump(gscv.best_estimator_, open("ada_imdb.model", 'wb'))


# Evaluate performance of each model on best parameters
print("Adaboost")
evaluation.evaluate_model(gscv.best_estimator_, X_train, y_train, X_test, y_test)



# Validation Curves
plt.title("Accuracy vs Number of estimators in Adaboost")

plt.xlim(50, 200)
plt.xlabel('Number of estimators')
plt.ylabel("Accuracy")

evaluation.plot_validation_curve(gscv.best_estimator_, "Adaboost", X_train, y_train, "n_estimators", np.linspace(50, 200, num=7, dtype=int), plot_number=1)
plt.legend(loc="best")

plt.show()
