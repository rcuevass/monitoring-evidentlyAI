import pandas as pd
from sklearn import datasets

from evidently.dashboard import Dashboard

# to check - jupyter nb example from documentation may be inaccurate
# from evidently.tabs import DriftTab
from evidently.tabs import DataDriftTab

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns=iris.feature_names)

