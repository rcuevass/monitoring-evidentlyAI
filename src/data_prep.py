import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, auc


def encode_features_ohe(feature_names: list, data_frame):
    for feature_name in feature_names:
        one_hot = pd.get_dummies(data_frame[feature_name])
        data_frame = data_frame.drop(feature_name, axis=1)
        data_frame = data_frame.join(one_hot)
    return data_frame


df = pd.read_csv('../data/input/archive/adult.csv')
df = encode_features_ohe(['race', 'gender', 'workclass'], df)


df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

x = df[['age', 'educational-num', 'capital-gain', 'capital-loss',
       'hours-per-week', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black',
        'Other', 'White', 'Female', 'Male', '?', 'Federal-gov', 'Local-gov',
        'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']]
y = df['income']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
