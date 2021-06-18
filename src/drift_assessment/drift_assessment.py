from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
from logs_generator.logging_script import get_log_object_named
from evidently.dashboard import Dashboard

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab
from evidently.tabs import DataDriftTab

from model_development.training import get_np_array_from_df
from data_read_prep.data_read import read_csv_select_type

log_ = get_log_object_named('drift')


def read_ref_prod_drift_score(path_pkl_model: str, path_ref_csv: str, path_prod_csv: str,
                              html_report_name: str = 'drift_report.html'):

    log_.info('reading csv file for reference data and selecting numeric features only, read from=%s', path_ref_csv)
    df_reference = read_csv_select_type(path_to_csv_file=path_ref_csv)
    log_.info('reading csv file for prod data and selecting numeric features only, read from=%s', path_prod_csv)
    df_prod = read_csv_select_type(path_to_csv_file=path_prod_csv)

    log_.info('Instantiating object for data drift report...')
    data_drift_report = Dashboard(tabs=[DataDriftTab])
    log_.info('performing calculations to generate report...')
    data_drift_report.calculate(reference_data=df_reference, production_data=df_prod, column_mapping=None)
    log_.info('saving data drift report to=%s', '../reports/' + html_report_name)
    data_drift_report.save('../reports/' + html_report_name)

    log_.info('Loading pickle file with trained data from =%s', path_pkl_model)
    clf_ = pickle.load(open(path_pkl_model, 'rb'))
    log_.info('Generating arrays of features and target for reference data from csv read from=%s', path_ref_csv)
    x_test_ref, y_test_ref = get_np_array_from_df(path_ref_csv)
    log_.info('Generating arrays of features and target for prod data from csv read from=%s', path_prod_csv)
    x_test_prod, y_test_prod = get_np_array_from_df(path_prod_csv)

    log_.info('Predicting on test reference data...')
    y_test_ref_hat = clf_.predict(x_test_ref)
    log_.info('Predicting on test prod data...')
    y_test_prod_hat = clf_.predict(x_test_prod)

    log_.info('computing accuracy and auc for test reference data...')
    acc_ref = accuracy_score(y_true=y_test_ref, y_pred=y_test_ref_hat)
    auc_ref = roc_auc_score(y_true=y_test_ref, y_score=y_test_ref_hat)

    log_.info('computing accuracy and auc for prod data...')
    acc_prod = accuracy_score(y_true=y_test_prod, y_pred=y_test_prod_hat)
    auc_prod = roc_auc_score(y_true=y_test_prod, y_score=y_test_prod_hat)

    log_.info('Accuracy on reference data =%s', str(acc_ref))
    log_.info('Accuracy on prod data =%s', str(acc_prod))
    log_.info('AUC on reference data =%s', str(auc_ref))
    log_.info('AUC on prod data =%s', str(auc_prod))
