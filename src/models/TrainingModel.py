import numpy as np
import pandas as pd
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class TrainingModel(object):

    def __init__(self, features):
        self.features = features

        self.X_training_df = pd.read_csv('data/interim/X_training.csv')
        self.y_training_df = pd.read_csv('data/interim/y_training.csv')

        self.X_test_df = pd.read_csv('data/interim/X_test.csv')
        self.y_test_df = pd.read_csv('data/interim/y_test.csv')
        print(self.X_test_df.shape)
        print(self.y_test_df.shape)

        self.X_training_df = self.X_training_df[self.features]
        self.X_test_df = self.X_test_df[self.features]

        self.X_test_df = self.X_test_df.replace(np.NaN, 0)
        self.X_test = self.X_test_df.values
        self.y_test = self.y_test_df.values.ravel()

        self.X_training = self.X_training_df.values
        self.y_training = self.y_training_df.values.ravel()

        self.logger = logging.getLogger(__name__)
        self.logger.info("features:", self.features)
        self.param_grid = {"hidden_layer_sizes": [(5), (10), (20), (40)],
                           "activation": ['identity', 'logistic', 'tanh', 'relu'],
                           "solver": ['lbfgs', 'sgd', 'adam'],
                           "alpha": [0.01, 0.001, 0.0001, 0.00001]}
        self.tree_param_grid = {"criterion": ["gini", "entropy"],
                                "splitter": ["best", "random"],
                                "min_samples_split": list(range(1, 50)),
                                }

    def training_model(self, model):
        if 'decision_tree' == model:
            self.decision_tree()
        elif 'multi_layer_perceptrons' == model:
            self.multi_layer_perceptrons()
        elif 'logistic_regression' == model:
            self.logistic_regression()
        elif 'grid_search_mlp' == model:
            self.grid_search()
        else:
            raise AttributeError(f'{model} is not supported. Insert other attribute i.e. decision_tree, '
                                 f'multi_layer_perceptrons, logistic_regression')

    def decision_tree(self):
        self.logger.info("Init Decision Tree training model")
        sample_split_range = list(range(1, 50))
        param_grid = dict(min_samples_split=sample_split_range)
        clf = DecisionTreeClassifier()

        # instantiate the grid
        grid = GridSearchCV(clf, self.tree_param_grid, cv=KFold(n_splits=5), scoring='accuracy')

        # fit the grid with data
        grid.fit(self.X_training, self.y_training)
        # Train Decision Tree Classifier

        # Predict the response for test dataset
        y_pred = grid.predict(self.X_test)

        print("Parameters: " + str(grid.best_params_))
        print("R^2 on training set: " + str(grid.score(self.X_training, self.y_training)))
        print("R^2 on test set: " + str(grid.score(self.X_test, self.y_test)))

        print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))
        print("Confusion Matrix:", metrics.confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", metrics.classification_report(self.y_test, y_pred))
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, y_pred))  # the most important
        print('Mean Squared Error:', metrics.mean_squared_error(self.y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)))

    def multi_layer_perceptrons(self):
        mlp_model = MLPClassifier(hidden_layer_sizes=(10), alpha=0.0001, activation='tanh',
                                 solver='lbfgs', random_state=1)
        mlp_model.fit(self.X_training, self.y_training)

        rng = np.random.RandomState(0)
        colors = rng.rand(2640)

        # Predict with non seen data.
        y_pred = mlp_model.predict(self.X_test)
        print(y_pred)
        print(self.y_test)
        print("M2E:", metrics.mean_squared_error(self.y_test, y_pred))
        print(metrics.confusion_matrix(self.y_test, y_pred))
        print(metrics.accuracy_score(self.y_test, y_pred))
        self.cross_val(mlp_model)

    def grid_search(self):
        self.logger.info("Init Multi Layer Perceptrons training model")

        mlp_model = MLPClassifier(random_state=1)
        ss = KFold(n_splits=5)

        grid = GridSearchCV(mlp_model, self.param_grid, cv=ss, n_jobs=-1).fit(self.X_training, self.y_training)

        mlp_model_tuned = grid.best_estimator_
        print("Parameters: " + str(grid.best_params_))
        print("R^2 on training set: " + str(mlp_model_tuned.score(self.X_training, self.y_training)))
        print("R^2 on test set: " + str(mlp_model_tuned.score(self.X_test, self.y_test)))

    def cross_val(self, model):
        from sklearn.model_selection import cross_val_score
        # If finish with _score higher better.
        scores = cross_val_score(model, self.X_training, self.y_training, cv=KFold(n_splits=5))
        print(scores)
        print(scores.mean())
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def logistic_regression(self):
        self.logger.info("Init Logistic Regression training model")
        clf = LogisticRegression()
        grid_values = {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25, 50, 75],
                       'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        grid = GridSearchCV(clf, param_grid=grid_values, scoring='accuracy')
        grid.fit(self.X_training, self.y_training)

        # Predict values based on new parameters
        y_pred_acc = grid.predict(self.X_test)

        mlp_model_tuned = grid.best_estimator_
        print("Parameters: " + str(grid.best_params_))
        print("R^2 on training set: " + str(mlp_model_tuned.score(self.X_training, self.y_training)))
        print("R^2 on test set: " + str(mlp_model_tuned.score(self.X_test, self.y_test)))

        # New Model Evaluation metrics
        print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred_acc))
        print("Confusion Matrix:", metrics.confusion_matrix(self.y_test, y_pred_acc))
        print("Classification Report:\n", metrics.classification_report(self.y_test, y_pred_acc))

        print('Accuracy Score : ' + str(metrics.accuracy_score(self.y_test, y_pred_acc)))
        print('Precision Score : ' + str(metrics.precision_score(self.y_test, y_pred_acc)))
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, y_pred_acc))  # the most important
        print('Mean Squared Error:', metrics.mean_squared_error(self.y_test, y_pred_acc))
        # print('Recall Score : ' + str(metrics.recall_score(self.y_test, y_pred_acc)))
        # print('F1 Score : ' + str(metrics.f1_score(self.y_test, y_pred_acc)))

        # Logistic Regression (Grid Search) Confusion matrix
        print(metrics.confusion_matrix(self.y_test, y_pred_acc))
