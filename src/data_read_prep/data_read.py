# to handle data frames
import pandas as pd


def read_csv_select_type(path_to_csv_file: str, data_type: str = 'number') -> pd.DataFrame:
    """
    Function that reads a csv file to a data frame and selects only certain data type
    :param path_to_csv_file: string indicated the path to and csv file
    :param data_type: string indicating the datatype to be kept
                      default set to numeric
    :return: data_frame: data frame capturing data
    """

    # read data from csv file
    data_frame = pd.read_csv(path_to_csv_file)
    # selecting data type from function argument
    data_frame = data_frame.select_dtypes(include=data_type)
    return data_frame
