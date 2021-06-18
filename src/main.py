from drift_assessment.drift_assessment import read_ref_prod_drift_score
from model_development.training import train_store_model
from logs_generator.logging_script import get_log_object_named
from os import path
import pandas as pd
from data_read_prep.data_prep import encode_features_ohe, force_drift
from sklearn.model_selection import train_test_split

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab

# instantiate log object
log_main = get_log_object_named('main')


def main():
    # training model and storing binary
    log_main.info('Training model and storing binary...')
    dict_acc_auc = train_store_model(path_csv_train='../data/input/train.csv',
                                     path_csv_test='../data/input/test_reference.csv',
                                     path_name_pkl_trained='../models/knn_clf.pkl')

    # read reference and prod data and score on both
    log_main.info('Reading reference and prod data and scoring...')
    read_ref_prod_drift_score(path_pkl_model='../models/knn_clf.pkl',
                              path_ref_csv='../data/input/test_reference.csv',
                              path_prod_csv='../data/input/test_prod_forced.csv')


if __name__ == '__main__':
    if path.exists('../data/input/train.csv') & path.exists('../data/input/test_reference.csv'):
        main()
    else:

        df = pd.read_csv('../data/input/archive/adult.csv')
        df = encode_features_ohe(['race', 'gender', 'workclass'], df)
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

        df_train, df_test = train_test_split(df, test_size=0.4, random_state=2021)
        df_test_ref, df_test_prod_unforced = train_test_split(df_test, test_size=0.5, random_state=2021)
        df_test_prod_forced = force_drift(df_test_prod_unforced, list_drift_num=['age'])

        df_train.to_csv('../data/input/train.csv', index=False)
        df_test_ref.to_csv('../data/input/test_reference.csv', index=False)
        df_test_prod_unforced.to_csv('../data/input/test_prod_unforced.csv', index=False)
        df_test_prod_forced.to_csv('../data/input/test_prod_forced.csv', index=False)
        main()
