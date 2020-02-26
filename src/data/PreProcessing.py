from src.visualization.visualize import *
from src.features.build_features import *
from src.data.Encoding import Encoding
from src.data.data_utils import save_data
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import MinMaxScaler


class PreProcessing(object):

    def __init__(self, input_path, output_path):
        self.logger = logging.getLogger(__name__)
        self.output_path = output_path
        self.raw_data = pd.read_csv(input_path)
        self.X = None
        self.y = None
        self.target_name = "made_deposit"
        self.feature_names = []

    def pre_processing(self):
        """
        Execute all task to pre_process data set.
        """
        self.data_info()
        self.summarize_data()
        self.clean_data()
        self.encoding()
        self.set_features()
        self.normalise()
        # filter_features(self.X, self.y)
        # self.visualize_data("numerical")

    def normalise(self):
        """

        """
        mm_scaler = MinMaxScaler()
        self.X = mm_scaler.fit_transform(self.X)

    def set_features(self):
        """
         Save features values in self.X and target value in self.y

        """
        self.feature_names = self.raw_data.columns[np.where(self.raw_data.columns != self.target_name)]
        self.raw_data.columns.get_loc(self.target_name)
        self.X = self.raw_data[self.feature_names].values
        self.y = self.raw_data[self.target_name].values.ravel()

    def clean_data(self):
        """
        Clean data function fix most common errors and fix them.

        """
        self.parse_column_type()
        self.clean_data_frame()
        self.fix_typos()

    def fix_typos(self):
        self.raw_data['last_contact'] = self.raw_data['last_contact'].replace({'cell': 'cellular'})
        self.raw_data['has_loan'] = self.raw_data['has_loan'].replace({'n': 'no'})

    def parse_column_type(self):
        # convert just columns "a" and "b"
        to_update = ["town", "country", "job", "married", "education", "defaulted?",
                     "housing", "has_loan", "last_contact_month", "poutcome",
                     "made_deposit"]
        self.raw_data[to_update] = self.raw_data[to_update].astype(str)
        self.logger.info("Shape: {}".format(self.raw_data.shape))

    def data_info(self):
        self.logger.info("Shape: {}".format(self.raw_data.shape))
        self.logger.info("Data Head:\n{}".format(self.raw_data.head(5)))
        self.logger.info(f'Columns type:\n{self.raw_data.columns.to_list()}')

    def clean_data_frame(self):
        self.logger.info("(rows, columns) prior to remove NULL/NaN: {}".format(self.raw_data.shape))
        # Drop any rows which have any nans
        self.raw_data.dropna()
        # Drop columns not used for analysis
        self.raw_data = self.raw_data.drop(['town', 'country', 'accountID'], axis=1)
        # Drop columns if have more than 70% of unknown value. In this case 'poutcome' attribute was removed
        self.raw_data = self.raw_data.loc[:, self.raw_data.isin(['unknown']).mean() < 0.7]
        # Drop row from incomplete data
        self.raw_data = self.raw_data[self.raw_data['last_contact_month'] != 'j']
        self.logger.info(f"(rows, columns) after clean Null/NaN, typos, missing data: {self.raw_data.shape}")

    def visualize_data(self):
        """Visualize data Categorical and Numerical.
        """
        df = self.raw_data
        for column in df.columns:
            if df.dtypes[column] == np.object:  # Categorical data
                h_bar_plot(df[column].value_counts())
            else:
                box_plot(df[column].describe())

    def summarize_data(self):
        """
        Summarize data info
        """
        df = self.raw_data
        for column in df.columns:
            print(column)
            if df.dtypes[column] == np.object:  # Categorical data
                print(df[column].value_counts())
            else:
                print(df[column].describe())
            print('\n')

    def set_train_test_data(self):
        """
        Split data and save as train and test.
        """
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=1)

        # Save encoded df
        self.save_slip_data(X_train, y_train, 'train.csv')
        self.save_slip_data(X_test, y_test, 'test.csv')

    def save_slip_data(self, X, y, file_name):
        """
        Given
        X and y parameters as Numpy array.

        When
        Transform to DataFrame

        Then
        Save as csv file
        """
        df_X = pd.DataFrame(data=X, columns=self.feature_names)
        df_y = pd.DataFrame(data=y, columns=[self.target_name])
        x_path = str(self.output_path + "/x_" + file_name)
        y_path = str(self.output_path + "/y_" + file_name)

        save_data(df_X, x_path)
        save_data(df_y, y_path)

        return df_X, df_y

    def encoding(self):
        """
        Encoding features and save it as csv file
        """
        self.raw_data = Encoding(self.raw_data).encode()
        save_data(self.raw_data, "data/processed/encode_data.csv")
