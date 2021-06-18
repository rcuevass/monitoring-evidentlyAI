import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
from logs_generator.logging_script import get_log_object_named

# instantiate log object
log_ = get_log_object_named('training')


def get_np_array_from_df(path_to_data: str) -> tuple:
    """
    Function that returns a tuple of numpy arrays: (array of features, array of targets), from
    path to data in csv format
    :param path_to_data: string capturing location of data in csv format
    :return: tuple of numpy arrays (features, target)
    """
    # read csv file to data frame
    log_.info('reading csv from=%s', path_to_data)
    data_frame = pd.read_csv(path_to_data)

    # extract features
    log_.info('extracting array of features')
    x = data_frame[['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'Amer-Indian-Eskimo',
                    'Asian-Pac-Islander', 'Black', 'Other', 'White', 'Female', 'Male', '?', 'Federal-gov', 'Local-gov',
                    'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']].values

    # extract target
    log_.info('extracting array of target')
    y = data_frame['income'].values

    # return tuple
    log_.info('returning tuple of features and target')
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
    log_.info('turning training csv into features and target=%s', path_csv_train)
    x_train, y_train = get_np_array_from_df(path_csv_train)
    log_.info('turning test csv into features and target=%s', path_csv_test)
    x_test, y_test = get_np_array_from_df(path_csv_test)

    # train and predict on both train and test
    log_.info('instantiating KNN classifier...')
    clf = KNeighborsClassifier(n_neighbors=num_neighbours)
    log_.info('fitting model...')
    clf.fit(x_train, y_train)
    log_.info('predicting for training set...')
    y_hat_train = clf.predict(x_train)
    log_.info('predicting for test set...')
    y_hat_test = clf.predict(x_test)

    # create empty dictionary for metrics of performance
    dict_acc_auc = dict()

    # get metrics for training ...
    log_.info('computing accuracy and auc for training...')
    acc_train = accuracy_score(y_true=y_train, y_pred=y_hat_train)
    auc_train = roc_auc_score(y_true=y_train, y_score=y_hat_train)

    # .. and for test
    log_.info('computing accuracy and auc for test...')
    acc_test = accuracy_score(y_true=y_test, y_pred=y_hat_test)
    auc_test = roc_auc_score(y_true=y_test, y_score=y_hat_test)

    # populate dictionary
    log_.info('populating dictionary with metrics of performance for training and test...')
    dict_acc_auc['acc_train'] = acc_train
    dict_acc_auc['auc_train'] = auc_train
    dict_acc_auc['acc_test'] = acc_test
    dict_acc_auc['auc_test'] = auc_test

    # displaying performance of trained model...
    log_.info('Metrics of performance for trained model=%s', str(dict_acc_auc))

    # save model to pkl file
    log_.info('saving trained models to pkl object in =%s', path_name_pkl_trained)
    pickle.dump(clf, open(path_name_pkl_trained, 'wb'))

    return dict_acc_auc
