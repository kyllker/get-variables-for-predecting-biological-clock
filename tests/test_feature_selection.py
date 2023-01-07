import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.FeatureSelection.feature_selection import FeatureSelection


class TestFeatureSelection:
    def setup_class(self):

        self.seed = 42
        self.feature_selection_object = FeatureSelection(self.seed)

        self.df = pd.DataFrame([[1, 1, 2], [1.001, 2, 4]], columns=['A', 'B', 'C'])

        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])

        self.iris = load_iris()

    def test_build_a_feature_selection_object(self):
        assert isinstance(self.feature_selection_object, FeatureSelection)

    def test_drop_little_variance_column(self):
        df_res = pd.DataFrame([[1, 2], [2, 4]], columns=['B', 'C'])
        assert self.feature_selection_object.drop_columns_little_variance(self.df, 0.05).equals(df_res)

    def test_feature_selection_with_ml_algorithms(self):
        x, y = load_iris(return_X_y=True)
        df_iris = pd.DataFrame(data=np.c_[x, y],
                               columns=self.iris['feature_names'] + ['target'])

        df_res = self.feature_selection_object.feature_selection_with_ml_algorithms(df_iris.iloc[:,:-1], y, 0.3)
        assert set(list(df_res)) == set(['petal length (cm)', 'petal width (cm)', 'sepal length (cm)'])