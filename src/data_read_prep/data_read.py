import pandas as pd


def read_csv_select_type(path_to_csv_file: str, data_type: str = 'number'):

    data_frame = pd.read_csv(path_to_csv_file)
    data_frame = data_frame.select_dtypes(include=data_type)
    return data_frame

