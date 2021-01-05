import evaluation
import matplotlib.pyplot as plt
import imdb
import pickle

X_train, y_train, X_test, y_test = imdb.fetch_data()

lr = pickle.load(open("lr_imdb.model", "rb"))
svm = pickle.load(open("svm_imdb.model", "rb"))
dt = pickle.load(open("dt_imdb.model", "rb"))
ada = pickle.load(open("ada_imdb.model", "rb"))
forest = pickle.load(open("forest_imdb.model", "rb"))
nb = pickle.load(open("nb_imdb.model", "rb"))
mlp = pickle.load(open("mlp_imdb.model", "rb"))


# Learning Curves

fig, axes = plt.subplots(2, 4)
ax = fig.add_subplot(111, frameon=False)
axes[1,3].remove()
evaluation.plot_learning_curve(lr, "Logistic Regression", X_train, y_train, axes[0,0])
evaluation.plot_learning_curve(svm, "SVM", X_train, y_train, axes[0,1])
evaluation.plot_learning_curve(nb, "Naive Bayes", X_train, y_train, axes[0,2])
evaluation.plot_learning_curve(mlp, "Multilayer Perceptron", X_train, y_train, axes[0,3])
evaluation.plot_learning_curve(dt, "Decision Tree", X_train, y_train, axes[1,0])
evaluation.plot_learning_curve(ada, "Adaboost", X_train, y_train, axes[1,1])
evaluation.plot_learning_curve(forest, "Random Forest", X_train, y_train, axes[1,2])

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')
#plt.tight_layout(w_pad=0.05)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.show()