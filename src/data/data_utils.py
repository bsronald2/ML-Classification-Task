import os


def save_data(df, file_name):
    if os.path.exists(file_name):
        os.remove(file_name)
    df.to_csv(file_name, index=False)