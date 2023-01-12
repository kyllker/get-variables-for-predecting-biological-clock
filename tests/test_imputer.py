import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)

from src.Preprocessors.imputer import imputer


class Testimputer:
    def setup_class(self):
        self.seed = 42
        self.imputer_object = imputer(self.seed)

        self.df = pd.DataFrame([[1, 1], [1, 2]], columns=['A', 'B'])

        self.df_na = pd.DataFrame([[1, np.nan], [1, 2]], columns=['A', 'B'])

        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])

        self.iris = load_iris()

    def test_build_an_imputer_object(self):
        assert isinstance(self.imputer_object, imputer)

    def test_check_there_are_na_values_false(self):
        assert not self.imputer_object.check_there_are_na_values(self.df)

    def test_check_there_are_na_values_true(self):
        assert self.imputer_object.check_there_are_na_values(self.df_na)

    def test_get_na_and_no_na_columns(self):
        assert self.imputer_object.get_na_and_no_na_columns(self.df_na)[0][0] == 'B' and \
               self.imputer_object.get_na_and_no_na_columns(self.df_na)[1][0] == 'A'

    def test_knn_classifier(self):
        X_train = self.iris.data[:145]
        y_train = self.iris.target[:145]
        X_predict = self.iris.data[145:]
        assert all(self.imputer_object.knn_classifier(X_train, y_train, X_predict) == [2, 2, 2, 2, 2])

    def test_knn_regressor(self):
        X_train = self.iris.data[:145]
        y_train = self.iris.target[:145]
        X_predict = self.iris.data[145:]
        assert all(self.imputer_object.knn_regressor(X_train, y_train, X_predict) == [2.0,  1.6, 2.0,  2.0,  1.8])

    def test__imputer_normalize_dataframe(self):
        df2 = pd.DataFrame([[1, 1], [2, 3]], columns=['B', 'C'])
        dataframe_drop_constant_columns = self.imputer_object.normalize_dataframe(df2)
        df_res = pd.DataFrame([[0, 0], [1, 1]], columns=['B', 'C'])
        assert np.array_equal(dataframe_drop_constant_columns.to_numpy(), df_res.to_numpy())

    def test_predict_with_knn_regressor(self):
        X, y = load_iris(return_X_y=True)
        mask = np.random.randint(0, 2, size=X.shape).astype(bool)
        X[mask] = np.nan
        df_iris = pd.DataFrame(data=np.c_[X, y], columns=self.iris['feature_names'] + ['target'])
        df_iris = df_iris.drop('target', axis=1)
        df_iris['sepal length (cm)'] = 1
        df_iris['petal length (cm)'] = 2
        df_iris_no_na = self.imputer_object.predict(df_iris, 'knn')
        assert df_iris_no_na.isna().sum().sum() == 0 and 1 < df_iris_no_na.iloc[0,1] < 3.9
