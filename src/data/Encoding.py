import pandas as pd


class Encoding(object):

    def __init__(self, df):
        self.yes_no_cat = {'no': 0, 'yes': 1}
        self.month_cat = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
                          'oct': 10, 'nov': 11, 'dec': 12}
        self.edu_cat = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
        self.df = df

    def encode(self):
        self.df.made_deposit = self.df['made_deposit'].map(self.yes_no_cat)
        self.df.has_loan = self.df['has_loan'].map(self.yes_no_cat)
        self.df.housing = self.df['housing'].map(self.yes_no_cat)
        self.df['defaulted?'] = self.df['defaulted?'].map(self.yes_no_cat)
        self.df.last_contact_month = self.df['last_contact_month'].map(self.month_cat)
        self.df.education = self.df['education'].map(self.edu_cat)

        # One-hot encode ordinal data
        self.df = pd.get_dummies(self.df, prefix="marital_status", columns=['married'])
        self.df = pd.get_dummies(self.df, columns=['last_contact'])
        self.df = pd.get_dummies(self.df, columns=['job'])

        return self.df
