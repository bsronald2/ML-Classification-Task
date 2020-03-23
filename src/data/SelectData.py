import logging
import pandas as pd
import numpy as np
from src.data.data_utils import save_data
from src.features.SelectFeatures import SelectFeatures
from sklearn.model_selection import train_test_split
from src.visualization.visualize import *
import os


class SelectData(object):

    def __init__(self, input_path, output_path):
        self.logger = logging.getLogger(__name__)
        self.output_path = output_path
        self.raw_data = pd.read_csv(input_path)
        self.logger.info(f"Raw Data Info {self.raw_data.shape}")
        save_image = eval(os.environ['SAVE_IMAGES'])
        if save_image:
            visualize.save_image_path = "reports/figures/data"
            visualize.hist_plot(self.raw_data, "raw_data")
        self.feature_names = None
        self.target_name = "made_deposit"
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_file = 'raw_train.csv'
        self.test_file = 'raw_test.csv'

        # set features and target
        self.set_features()

    def set_train_test_data(self):
        """
                Split data and save as train and test.
        """
        # split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,
                                                                                random_state=1)

        self.logger.info(f'Shape Training Set X and y {self.X_train.shape}  {self.y_train.shape}')
        self.logger.info(f'Shape Training Set X and y {self.X_test.shape}  {self.y_test.shape}')
        # Save raw test and training data as df
        self.save_slip_data(self.get__training_features(), self.get__training_target(), 'raw_train.csv')
        self.save_slip_data(self.get__test_features(), self.get__test_target(), 'raw_test.csv')

    def set_features(self):
        self.feature_names = self.raw_data.columns[np.where(self.raw_data.columns != self.target_name)]
        self.raw_data.columns.get_loc(self.target_name)
        self.X = self.raw_data[self.feature_names].values
        self.y = self.raw_data[self.target_name].values.ravel()

    def save_slip_data(self, X, y, file_name):
        """
        Given
        X and y parameters as Numpy array.

        When
        Transform to DataFrame

        Then
        Save as csv file
        """
        x_path = str(self.output_path + "/x_" + file_name)
        y_path = str(self.output_path + "/y_" + file_name)

        save_data(X, x_path)
        save_data(y, y_path)

    def get__training_features(self):

        return pd.DataFrame(data=self.X_train, columns=self.feature_names).infer_objects()

    def get__test_features(self):

        return pd.DataFrame(data=self.X_test, columns=self.feature_names).infer_objects()

    def get__training_target(self):

        return pd.DataFrame(data=self.y_train, columns=[self.target_name]).infer_objects()

    def get__test_target(self):

        return pd.DataFrame(data=self.y_test, columns=[self.target_name]).infer_objects()

    def select_data(self):
        self.set_train_test_data()
