
import matplotlib.pyplot as plt
import imdb
import evaluation
import pickle

# get the train-test data
X_train, y_train, X_test, y_test = imdb.fetch_data()

lr = pickle.load(open("lr_imdb.model", "rb"))
evaluation.evaluate_model(lr, X_train, y_train, X_test, y_test)


svm = pickle.load(open("svm_imdb.model", "rb"))
evaluation.evaluate_model(svm, X_train, y_train, X_test, y_test)

"""

# Validation Curves
# Setup plot

plt.title("Accuracy vs tol for Logistic regression and SVM")

plt.xlim(10, 0.0001)
plt.xlabel('tol')
plt.ylabel("Accuracy")

plt.xscale("log")

evaluation.plot_validation_curve(lr, "Logistic Regression", X_train, y_train, "tol", [10, 1, 0.1, 0.01, 0.001, 0.0001], plot_number=1)
evaluation.plot_validation_curve(svm, "SVM", X_train, y_train, "tol", [10, 1, 0.1, 0.01, 0.001, 0.0001], plot_number=3)
plt.legend(loc="best")

plt.show()
"""

