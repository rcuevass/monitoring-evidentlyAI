import pandas as pd


def encode_features_ohe(feature_names: list, data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Function that performs one hot encoding to a set of features provided as a list
    :param feature_names: list capturing the features to be encoded
    :param data_frame: dataframe capturing the data
    :return: data_frame: input data_frame after encoding of features
    """
    for feature_name in feature_names:
        try:
            one_hot = pd.get_dummies(data_frame[feature_name])
            data_frame = data_frame.drop(feature_name, axis=1)
            data_frame = data_frame.join(one_hot)
        except:
            print('Feature ', feature_name, 'may not be part of dataset')

    return data_frame


def force_drift(data_frame, list_drift_num: list) -> pd.DataFrame:
    """
    Function that adds a ad-hoc number set by the user to numerical features
    :param data_frame: data frame capturing the data that captures the data to be modified
    :param list_drift_num: list capturing the numerical features to be modified
    :return: data_frame: data frame with modified distribution
    """

    data_frame_out = data_frame.copy()
    for feat_num in list_drift_num:
        # get mean and std of column
        mean_ = data_frame[feat_num].mean()
        std_ = data_frame[feat_num].std()
        data_frame_out[feat_num] = round(mean_-20, 0)

    return data_frame_out
