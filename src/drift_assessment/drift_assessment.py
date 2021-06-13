from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

from evidently.dashboard import Dashboard

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab
from evidently.tabs import DataDriftTab

from model_development.training import get_np_array_from_df
from data_read_prep.data_read import read_csv_select_type


def read_ref_prod_drift_score(path_pkl_model: str, path_ref_csv: str, path_prod_csv: str,
                              html_report_name: str = 'drift_report.html'):
    df_reference = read_csv_select_type(path_to_csv_file=path_ref_csv)
    df_prod = read_csv_select_type(path_to_csv_file=path_prod_csv)

    data_drift_report = Dashboard(tabs=[DataDriftTab])
    data_drift_report.calculate(reference_data=df_reference, production_data=df_prod, column_mapping=None)
    data_drift_report.save('../reports/' + html_report_name)

    clf_ = pickle.load(open(path_pkl_model, 'rb'))
    x_test_ref, y_test_ref = get_np_array_from_df(path_ref_csv)
    x_test_prod, y_test_prod = get_np_array_from_df(path_prod_csv)

    y_test_ref_hat = clf_.predict(x_test_ref)
    y_test_prod_hat = clf_.predict(x_test_prod)

    acc_ref = accuracy_score(y_true=y_test_ref, y_pred=y_test_ref_hat)
    auc_ref = roc_auc_score(y_true=y_test_ref, y_score=y_test_ref_hat)

    acc_prod = accuracy_score(y_true=y_test_prod, y_pred=y_test_prod_hat)
    auc_prod = roc_auc_score(y_true=y_test_prod, y_score=y_test_prod_hat)

    print(acc_ref, auc_ref)
    print(acc_prod, auc_prod)
