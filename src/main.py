import pandas as pd
from sklearn import datasets

from evidently.dashboard import Dashboard

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab
from evidently.tabs import DataDriftTab

df_reference = pd.read_csv('../data/input/test_reference.csv')
df_reference = df_reference.select_dtypes(include='number')
df_prod = pd.read_csv('../data/input/test_prod.csv')
df_prod = df_prod.select_dtypes(include='number')

data_drift_report = Dashboard(tabs=[DataDriftTab])
data_drift_report.calculate(reference_data=df_reference, production_data=df_prod, column_mapping=None)
data_drift_report.save('../reports/drift_report.html')


