from src.visualization.visualize import *
from src.features.build_features import *
from src.data.Encoding import Encoding
from src.data.data_utils import *
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class PrePProcessing(object):

    def __init__(self, X, y, data_type):
        self.logger = logging.getLogger(__name__)
        self.X_df = X
        self.y_df = y
        self.X = None
        self.y = None
        self.to_drop = None
        self.path_save_image = "reports/figures/data"
        self.data_type = data_type
        self.save_image = eval(os.environ['SAVE_IMAGES'])

        # Drop columns not used for analysis
        self.X_df = self.X_df.drop(['town', 'country', 'accountID', 'poutcome'], axis=1)
        # self.X_df = self.X_df.drop(['town', 'country', 'accountID'], axis=1)

    def pre_processing(self):
        """
        Execute all task to pre_process data set.
        """
        self.encoding()
        self.normalise()
        self.save_data()

        return self.X_df, self.y_df

    def clean_data(self):
        self.clean_features_data()
        self.clean_target_data()

    def normalise(self):
        """
        Normalise data.
        """
        self.X = self.X_df.values
        self.y = self.y_df.values.ravel()
        self.X = normalise(self.X)
        self.X_df = pd.DataFrame(data=self.X, columns=self.X_df.columns).infer_objects()
        # self.y = normalise(self.y)

    def clean_features_data(self):
        """
        Clean data function fix most common errors and fix them.
        """
        self.clean_features()
        self.fix_features_typos()
        self.group_similar_classes()

    def group_similar_classes(self):
        """
        Group similar classes into one.
        """
        self.X_df["job"].replace(["admin.", "management"], 'management', inplace=True)
        self.X_df["job"].replace(["blue-collar.", "technician"], 'blue-collar', inplace=True)
        self.X_df["job"].replace(["self-employed", "entrepreneur"], 'self-employed', inplace=True)
        self.X_df["married"].replace(["single", "divorced"], 'single', inplace=True)

        if self.save_image:
            visualize.save_image_path = f'{self.path_save_image}/features'
            visualize.h_bar_plot(self.X_df["job"].value_counts(), "job_grouped_h_bar")
            visualize.h_bar_plot(self.X_df["married"].value_counts(), "married_grouped_h_bar")

    def clean_features(self):
        self.logger.info("(rows, columns) prior to remove NULL/NaN: {}".format(self.X_df.shape))
        # Drop any rows which have any nans
        self.X_df.dropna()
        # Drop columns if have more than 70% of unknown value. In this case 'poutcome' attribute was removed
        # self.X_df = self.X_df.loc[:, self.X_df.isin(['unknown']).mean() < 0.7]
        # Drop row from incomplete data
        self.to_drop = self.X_df['last_contact_month'].index[self.X_df['last_contact_month'] == 'j'].tolist()
        self.X_df = self.X_df[self.X_df['last_contact_month'] != 'j']
        self.logger.info(f"(rows, columns) after clean Null/NaN, typos, missing data: {self.X_df.shape}")

    def fix_features_typos(self):
        self.X_df['last_contact'] = self.X_df['last_contact'].replace({'cell': 'cellular'})
        self.X_df['has_loan'] = self.X_df['has_loan'].replace({'n': 'no'})

    def parse_column_type(self, df, to_update, data_type):
        """This method parse columns to an specific data type to objects"""
        for column in to_update:
            df[column] = df[column].astype(data_type)
        self.logger.info("Shape: {}".format(df.dtypes))

        return df

    def data_info(self):
        self.logger.info("Data Head X:\n{}".format(self.X_df.head(5)))
        self.logger.info("Data Head y:\n{}".format(self.y_df.head(5)))
        self.logger.info(f'X Columns type:\n{self.X_df.dtypes}')
        self.logger.info(f'y Columns type:\n{self.y_df.dtypes}')
        self.summarize_data(self.X_df)
        self.summarize_data(self.y_df)
        if self.save_image:
            visualize.visualize_data(self.X_df, f'{self.path_save_image}/features')
            visualize.visualize_data(self.y_df, f'{self.path_save_image}/target')
            visualize.correlation(self.X_df, f'{self.path_save_image}/features/correlation')
            visualize.scatter_matrix(self.X_df, f'{self.path_save_image}/features/scatter_plot_matrix')

    def summarize_data(self, df):
        """
        Summarize data info
        """
        print("--------------Summarize Data Set Data Type--------------")
        for index, value in df.dtypes.iteritems():
            if value == object:  # Categorical data
                print(df[index].value_counts(normalize=True) * 100) # print count by percentage
            else:
                print(df[index].describe())
            print("\n")
        print("--------------------------------------------------------")

    def encoding(self):
        """
        Encoding features and save it as csv file
        """
        self.X_df, self.y_df = Encoding(self.X_df, self.y_df).encode()
        # if self.save_image:
        #     visualize.correlation(self.X_df, f'{self.path_save_image}/features/correlation_encoded')
        visualize.correlation(self.X_df, f'{self.path_save_image}/features/correlation_encoded')
        # visualize.scatter_matrix(self.X_df, f'{self.path_save_image}/features/scatter_encoded')

    def clean_target_data(self):
        self.y_df = self.y_df.drop(self.to_drop)

    def save_data(self):
        save_data(self.X_df, f"data/interim/X_{self.data_type}.csv")
        save_data(self.y_df, f"data/interim/y_{self.data_type}.csv")
