from drift_assessment import read_ref_prod_drift_score

from evidently.dashboard import Dashboard

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab
from evidently.tabs import DataDriftTab


read_ref_prod_drift_score(path_pkl_model='../models/knn_clf.pkl',
                          path_ref_csv='../data/input/test_reference.csv',
                          path_prod_csv='../data/input/test_prod_forced.csv',
                          html_report_name='drift_report.html')
