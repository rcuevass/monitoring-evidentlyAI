import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def get_np_array_from_df(path_to_data: str) -> tuple:
    data_frame = pd.read_csv(path_to_data)
    x = data_frame[['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'Amer-Indian-Eskimo',
                    'Asian-Pac-Islander', 'Black', 'Other', 'White', 'Female', 'Male', '?', 'Federal-gov', 'Local-gov',
                    'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']]

    y = data_frame['income']

    return x, y


x_train, y_train = get_np_array_from_df('../data/input/train.csv')
x_test, y_test = get_np_array_from_df('../data/input/test_reference.csv')


knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(x_train, y_train)
y_hat = knn_clf.predict(x_test)

acc_value = accuracy_score(y_true=y_test, y_pred=y_hat)
auc_value = roc_auc_score(y_true=y_test, y_score=y_hat)
print(acc_value)
print(auc_value)




