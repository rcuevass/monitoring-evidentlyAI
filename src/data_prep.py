import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


def encode_features_ohe(feature_names: list, data_frame):
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


def force_drift(data_frame, dict_drift_num: dict):
    """
    Function that adds a ad-hoc number set by the user to numerical features
    :param data_frame: data frame capturing the data that captures the data to be modified
    :param dict_drift_num: dictionary capturing the numerical features to be shifted as keys and amount by which
                            the shift will take place as values. dict[num_feat] = shift_amount
    :return: data_frame: data frame with modified distribution
    """

    data_frame_out = data_frame.copy()
    for feat_num in dict_drift_num.keys():
        val_to_add = dict_drift_num[feat_num]
        data_frame_out[feat_num] += val_to_add

    return data_frame_out


df = pd.read_csv('../data/input/archive/adult.csv')
df = encode_features_ohe(['race', 'gender', 'workclass'], df)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})


df_train, df_test = train_test_split(df, test_size=0.4, random_state=2021)
df_test_ref, df_test_prod_unforced = train_test_split(df_test, test_size=0.5, random_state=2021)
df_test_prod_forced = force_drift(df_test_prod_unforced, dict_drift_num={'age': 3, 'hours-per-week': -2})

df_train.to_csv('../data/input/train.csv', index=False)
df_test_ref.to_csv('../data/input/test_reference.csv', index=False)
df_test_prod_unforced.to_csv('../data/input/test_prod_unforced.csv', index=False)
df_test_prod_forced.to_csv('../data/input/test_prod_forced.csv', index=False)


print(df.shape)
print(df_train.shape)
print(df_test.shape)
print(df_test_prod_unforced.shape)
print(df_test_ref.shape)


