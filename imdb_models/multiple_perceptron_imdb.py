
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import imdb
import evaluation
import pickle

# get the train-test data
X_train, y_train, X_test, y_test = imdb.fetch_data()

# Get best hyper parameters by grid search

parameter = {
    'hidden_layer_sizes': [(50,), (100,)],
    'alpha': [0.0001, 0.01]
}



gscv = GridSearchCV(MLPClassifier(), parameter, n_jobs=-1, verbose=10, cv=5)
gscv.fit(X_train, y_train)


best_para = gscv.best_estimator_.get_params()
for param_name in sorted(parameter.keys()):
    print('%s %r' % (param_name, best_para[param_name]))

print("Validation Acc:", gscv.best_score_)
pickle.dump(gscv.best_estimator_, open("mlp_imdb.model", 'wb'))

# Evaluate performance of each model on best parameters
print("Multilayer perceptron")
evaluation.evaluate_model(gscv.best_estimator_, X_train, y_train, X_test, y_test)



# Validation Curves
#evaluation.plot_validation_curve(gscv.best_estimator_, "title", X_train, y_train, "alpha", [0, 0.0001, 0.001, 0.01, 0.1, 1])
#plt.show()
