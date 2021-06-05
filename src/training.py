import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle


def get_np_array_from_df(path_to_data: str) -> tuple:
    """
    Function that returns a tuple of numpy arrays: (array of features, array of targets), from
    path to data in csv format
    :param path_to_data: string capturing location of data in csv format
    :return: tuple of numpy arrays (features, target)
    """
    # read csv file to data frame
    data_frame = pd.read_csv(path_to_data)

    # extract features
    x = data_frame[['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'Amer-Indian-Eskimo',
                    'Asian-Pac-Islander', 'Black', 'Other', 'White', 'Female', 'Male', '?', 'Federal-gov', 'Local-gov',
                    'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']].values

    # extract target
    y = data_frame['income'].values

    # return tuple
    return x, y


def train_store_model(path_csv_train: str, path_csv_test: str, path_name_pkl_trained: str,
                      num_neighbours: int = 5) -> dict:
    """
    Function that trains a classifier and stores it as a pickle file, it return a tuple capturing accuracy
    and area under the curve
    :param path_csv_train: string indicating name and location of training dataset
    :param path_csv_test: string indicating name and location of test dataset
    :param path_name_pkl_trained: string capturing name and location where model will be stored as pkl
    :param num_neighbours: integer capturing number of neighbours; defaulted to 5
    :return: dictionary capturing accuracy and area under the curve for training and test
    """

    # features and target for training and test from path to csv
    x_train, y_train = get_np_array_from_df(path_csv_train)
    x_test, y_test = get_np_array_from_df(path_csv_test)

    # train and predict on both train and test
    clf = KNeighborsClassifier(n_neighbors=num_neighbours)
    clf.fit(x_train, y_train)
    y_hat_train = clf.predict(x_train)
    y_hat_test = clf.predict(x_test)

    # create empty dictionary for metrics of performance
    dict_acc_auc = dict()

    # get metrics for training ...
    acc_train = accuracy_score(y_true=y_train, y_pred=y_hat_train)
    auc_train = roc_auc_score(y_true=y_train, y_score=y_hat_train)

    # .. and for test
    acc_test = accuracy_score(y_true=y_test, y_pred=y_hat_test)
    auc_test = roc_auc_score(y_true=y_test, y_score=y_hat_test)

    # populate dictionary
    dict_acc_auc['acc_train'] = acc_train
    dict_acc_auc['auc_train'] = auc_train
    dict_acc_auc['acc_test'] = acc_test
    dict_acc_auc['auc_test'] = auc_test

    # save model to pkl file
    pickle.dump(clf, open(path_name_pkl_trained, 'wb'))

    return dict_acc_auc


dict_performance = train_store_model(path_csv_train='../data/input/train.csv',
                                     path_csv_test='../data/input/test_reference.csv',
                                     path_name_pkl_trained='../models/knn_clf.pkl')

print(dict_performance)
