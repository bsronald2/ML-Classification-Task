from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import math
import numpy as np
from numpy.random import seed
from numpy.random import randint
import os


class SelectFeatures(object):

    def __init__(self, X, y):
        self.X = X.values
        self.y = y.values.ravel()
        self.X_df = X
        self.y_df = y
        self.features_path = 'references/features'
        self.logger = logging.getLogger(__name__)

    def tree_based_model(self):
        et_model = ExtraTreesClassifier(n_estimators=50)
        et_model = et_model.fit(self.X, self.y)
        self.logger.info("--------------TREE-------------------")
        # extract feature importances
        feat_importances = et_model.feature_importances_
        self.logger.info("Importances: " + str(feat_importances))

        # this is how we use the importances to select the features in the data
        # in this case, it will use the mean importance as a threshold for selection
        model = SelectFromModel(et_model, prefit=True)
        X_new = model.transform(self.X)
        self.logger.info("Updated shape: " + str(X_new.shape))

        # we can also plot the importances graphically...
        # for this we match the importances with the names of the features
        # in a pandas series
        feat_ranks = pd.Series(feat_importances, index=self.X_df.columns)
        feat_ranks.sort_values(inplace=True)
        # the higher, the more important the feature
        self.logger.info("Ranks: ")
        self.logger.info(feat_ranks)  # as text
        # plot = feat_ranks.plot(kind="barh")  # as plot
        # plt.show()

    def logistic_regression(self):
        self.logger.info("\n--------------Recursive Feature Elimination-------------------")
        lgr_model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=1000)
        rfe = RFE(lgr_model, 21)
        fit = rfe.fit(self.X, self.y)

        print("Number of features selected: " + str(fit.n_features_))
        print("Selected Features: " + str(fit.support_))

        # use a numpy array as that can be filtered by a list of booleans
        features_selected = np.array(self.X_df.columns)[fit.support_]
        print("Wrapper Selected Feature Names: " + str(features_selected))
        SelectFeatures.save_features(features_selected, f'{self.features_path}/wrapper_feature_selection')

        return features_selected

        # data = np.load('mat.npz', allow_pickle=True)
        # print(data)
        # print(data['features'])
        # print(type(data['features']))

        # Selected
        # Feature
        # Names: ['defaulted?' 'housing' 'has_loan' 'marital_status_married'
        #         'last_contact_cellular' 'last_contact_unknown' 'job_entrepreneur'
        #         'job_housemaid' 'job_retired' 'job_student']

        # ['defaulted?' 'housing' 'has_loan' 'campaign' 'previous'
        #  'marital_status_divorced' 'marital_status_married'
        #  'marital_status_single' 'last_contact_cellular' 'last_contact_telephone'
        #  'last_contact_unknown' 'job_admin.' 'job_blue-collar' 'job_entrepreneur'
        #  'job_housemaid' 'job_retired' 'job_self-employed' 'job_services'
        #  'job_student' 'job_technician' 'job_unemployed']

        # use a numpy array as that can be filtered by a list of booleans

    def wrapper_customised(self):
        mlp_model = MLPRegressor(hidden_layer_sizes=(5), alpha=0.001, batch_size='auto', solver='lbfgs',
                                 learning_rate='constant', learning_rate_init=0.01, max_iter=1000, random_state=1)

        # we will use an array of booleans, one for each feature, to select the features
        # this is done using X_train[:,selected] below
        # True means a feature is selected, False means it is not selected
        # You can see how to selected columns in a numpy array this way here:
        # https://stackoverflow.com/questions/19984102/select-elements-of-numpy-array-via-boolean-mask-array

        # as a starting point, make a numpy array of 10 True values (i.e. select all features)
        selected = np.full(self.X.shape[1], True)

        # initialise "best" to negative infinity, so on the first iteration we'll make the
        # "all features" set the best
        best_score = -math.inf
        best_selected = selected
        self.logger.info(best_score)
        # the random number generator needs a seed so it will produce the same sequence every time
        # this makes it possible to replicate the results later, and so also makes debugging simpler
        seed(10)

        # we will repeat this for 100 iterations
        for i in range(100):

            # evaluate current
            score = cross_val_score(mlp_model, self.X[:, selected], self.y, cv=3,
                                    scoring='neg_mean_squared_error').mean()

            # if current is best so far, store it
            if score > best_score:
                best_score = score
                best_selected = selected.copy()

            # reset "selected" array to the best found so far
            selected = best_selected.copy()

            # make a new feature set to test
            # by choosing a features at random and adding/removing it
            # (depending on whether it was already selected)
            to_change = randint(len(selected))
            selected[to_change] = not selected[to_change]

            # with a 50% probability, add/remove another feature...
            if randint(2) == 0:
                to_change = randint(len(selected))
                selected[to_change] = not selected[to_change]

        # once we're done, output the results
        self.logger.info("Best score found:" + str(best_score))
        features_selected = self.X_df.columns[best_selected]
        self.logger.info("Using features:" + str(features_selected))
        SelectFeatures.save_features(features_selected, f'{self.features_path}/wrapper_customised_feature_selection', )

        return features_selected
        # data = np.load('mat.npz')
        # print(type(data['features']))
        # print(data['features'])

        # Output
        # ['education', 'defaulted?', 'housing', 'has_loan',
        #  'last_contact_duration_s', 'campaign', 'previous',
        #  'marital_status_married', 'marital_status_single',
        #  'last_contact_cellular', 'last_contact_unknown', 'job_admin.',
        #  'job_blue-collar', 'job_entrepreneur', 'job_management', 'job_retired',
        #  'job_self-employed', 'job_services', 'job_student', 'job_technician',
        #  'job_unemployed'],

        # Education na
        # Using
        # features: Index(['defaulted?', 'housing', 'has_loan', 'cc_tr', 'last_contact_day',
        #                  'last_contact_month', 'last_contact_duration_s', 'campaign',
        #                  'days_since_last_contact', 'previous', 'marital_status_married',
        #                  'marital_status_single', 'last_contact_cellular',
        #                  'last_contact_telephone', 'last_contact_unknown', 'job_admin.',
        #                  'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_retired',
        #                  'job_self-employed', 'job_student', 'job_technician', 'job_unemployed'],
        #                 dtype='object')

    def embedded(self):
        reg = LassoCV()
        reg.fit(self.X, self.y)
        self.logger.info("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        self.logger.info("Best score using built-in LassoCV: %f" % reg.score(self.X, self.y))
        coef = pd.Series(reg.coef_, index=self.X_df.columns)
        self.logger.info("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
            sum(coef == 0)) + " variables")

        imp_coef = coef.sort_values()
        plt.rcParams['figure.figsize'] = (8.0, 10.0)
        imp_coef.plot(kind="barh")
        plt.title("Feature importance using Lasso Model")
        plt.show()

    @staticmethod
    def save_features(x, file):
        if os.path.exists(file):
            os.remove(file)
        np.savez(file, features=x)

    @staticmethod
    def load_features(file):
        """Return a numpy array"""
        data = np.load(file, allow_pickle=True)
        return data
