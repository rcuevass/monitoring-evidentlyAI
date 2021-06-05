import pandas as pd
from sklearn import datasets

from evidently.dashboard import Dashboard

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab
from evidently.tabs import DataDriftTab

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)

iris_data_drift_report = Dashboard(tabs=[DataDriftTab])
iris_data_drift_report.calculate(iris_frame[:100], iris_frame[100:], column_mapping=None)

iris_data_drift_report.save('../reports/my_report.html')


