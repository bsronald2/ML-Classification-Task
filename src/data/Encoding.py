import pandas as pd
import numpy as np


class Encoding(object):

    def __init__(self, X, y):
        self.yes_no_cat = {'no': 0, 'yes': 1}
        self.month_cat = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                          'oct': 10, 'nov': 11, 'dec': 12}
        self.edu_cat = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
        self.p_outcome = {}
        self.X = X
        self.y = y

    def encode(self):
        self.y.made_deposit = self.y['made_deposit'].map(self.yes_no_cat)
        self.X.has_loan = self.X['has_loan'].map(self.yes_no_cat)
        self.X.housing = self.X['housing'].map(self.yes_no_cat)
        self.X['defaulted?'] = self.X['defaulted?'].map(self.yes_no_cat)
        self.X.last_contact_month = self.X['last_contact_month'].map(self.month_cat)
        # self.X.education = self.X['education'].map(self.edu_cat)

        # One-hot encode ordinal data
        self.X = pd.get_dummies(self.X, prefix="marital_status", columns=['married'])
        self.X = pd.get_dummies(self.X, columns=['last_contact', 'job', 'education'])
        # self.X = pd.get_dummies(self.X, columns=['last_contact', 'job'])

        return self.X, self.y
