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


def get_categorical_data(X):
    # Get list of categorical variables
    s = (X.dtypes == 'object')
    object_cols = list(s[s].index)

    print("Categorical variables:")
    print(object_cols)

    return object_cols


def get_unique_entries(X):
    # Get number of unique entries in each column with categorical data
    object_cols = get_categorical_data(X)
    object_nunique = list(map(lambda col: X[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    # Print number of unique entries by column, in ascending order
    unique_entries = sorted(d.items(), key=lambda x: x[1])
    print(unique_entries)

    return unique_entries
