import os
from sklearn.preprocessing import MinMaxScaler


def save_data(df, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    df.to_csv(file_name, index=False)


def delete_files(my_dir):
    file_list = [f for f in os.listdir(my_dir)]
    for file in file_list:
        os.remove(os.path.join(my_dir, file))


def normalise(X):
    """
        Normalise data.
        """
    mm_scaler = MinMaxScaler()

    return mm_scaler.fit_transform(X)

