
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve



from time import time

def evaluate_model(model, X_train, y_train, X_test, y_test):

  # Time to train

  t0 = time()
  model.fit(X_train, y_train)
  print("Time to train:", time() - t0)

  t0 = time()
  pred = model.predict(X_test)
  print("Time to predict:", time() - t0)

  # Accuracy
  print("Accuracy:", metrics.accuracy_score(y_test, pred))

  print("Precision:", metrics.precision_score(y_test, pred, average='macro'))
  print("Recall:", metrics.recall_score(y_test, pred, average='macro'))
  print("F1 Score:", metrics.f1_score(y_test, pred, average='macro'))

  #plt.matshow(metrics.confusion_matrix(y_test, pred))
  #plt.title('Confusion matrix')
  #plt.colorbar()
  #plt.show()



def plot_learning_curve(estimator, title, X, y,axis, ylim=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 2 plots: the test and training learning curve, the training
    samples vs fit times curve

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    Based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)

    #axis.set_xlabel("Training examples")
    #axis.set_ylabel("Accuracy")
    axis.grid()




    train_sizes, train_scores, test_scores= learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=train_sizes, verbose=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axis.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="C0")
    axis.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="C1")
    axis.plot(train_sizes, train_scores_mean, 'o-', color="C0", label="Training score")
    axis.plot(train_sizes, test_scores_mean, 'o-', color="C1", label="Cross-validation score")

    # Plot n_samples vs fit_times


    return axis

def plot_validation_curve(estimator, title, X, y, param_name, param_range, plot_number=1):



    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv=5, verbose=10)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color=("C" + str(plot_number - 1)))
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color=("C" + str(plot_number)))
    plt.plot(param_range, train_scores_mean, 'o-', color=("C" + str(plot_number - 1)),
                 label=(title + " Training score"))
    plt.plot(param_range, test_scores_mean, 'o-', color=("C" + str(plot_number)),
                 label=(title + " Cross-validation score"))
