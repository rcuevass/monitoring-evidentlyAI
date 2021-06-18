from drift_assessment.drift_assessment import read_ref_prod_drift_score
from model_development.training import train_store_model
from logs_generator.logging_script import get_log_object_named

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab

# instantiate log object
log_main = get_log_object_named('main')

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
