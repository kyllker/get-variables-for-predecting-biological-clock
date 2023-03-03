import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)

from src.Preprocessors.imputer import Imputer


class TestImputer:
    def setup_class(self):
        self.seed = 42
        self.imputer_object = Imputer(self.seed)

        self.df = pd.DataFrame([[1, 1], [1, 2]], columns=['A', 'B'])

        self.df_na = pd.DataFrame([[1, np.nan], [1, 2]], columns=['A', 'B'])

        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])

        self.iris = load_iris()

    def test_build_an_imputer_object(self):
        assert isinstance(self.imputer_object, Imputer)

    def test_check_there_are_na_values_false(self):
        assert not self.imputer_object.check_there_are_na_values(self.df)

    def test_check_there_are_na_values_true(self):
        assert self.imputer_object.check_there_are_na_values(self.df_na)

    def test_get_na_columns_when_there_are_not(self):
        assert len(self.imputer_object.get_na_columns(self.df)) == 0

    def test_get_na_columns_when_there_are(self):
        assert self.imputer_object.get_na_columns(self.df_na)[0] == 'B'

    def test_get_no_na_columns_when_there_are_not(self):
        assert len(self.imputer_object.get_no_na_columns(self.df)) == self.df.shape[1]

    def test_get_no_na_columns_when_there_are(self):
        assert self.imputer_object.get_no_na_columns(self.df_na)[0] == 'A'

    def test_mean_or_mode_classifier(self):
        df_mode = pd.DataFrame([['a', 'b', 'f'], ['c', 'd', 'f']], columns=['A', 'B', 'C'])
        y_train = ['x', 'x', 'c']
        df_predict = pd.DataFrame([['a', 'b', 'f'], ['c', 'd', np.nan]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.mean_or_mode_classifier(df_mode, y_train, df_predict, 'C')
        assert list_res == ['x', 'x']

    def test_mean_or_mode_regressor(self):
        df_mode = pd.DataFrame([['a', 'b', 'f'], ['c', 'd', 'f']], columns=['A', 'B', 'C'])
        y_train = [1, 2, 3]
        df_predict = pd.DataFrame([['a', 'b', 'f'], ['c', 'd', np.nan]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.mean_or_mode_regressor(df_mode, y_train, df_predict, 'C')
        assert list_res == [2, 2]

    def test_linear_classifier(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8]], columns=['A', 'B', 'C'])
        y_train = ['a', 'b']
        df_predict = pd.DataFrame([[1.1, 0.9, 1], [2.1, 2.05, 2]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.linear_classifier(df_mode, y_train, df_predict, 'D')
        assert set(list_res) == set(['a', 'b'])

    def test_linear_regressor(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8]], columns=['A', 'B', 'C'])
        y_train = [8, 9]
        df_predict = pd.DataFrame([[1.1, 0.9, 1], [2.1, 2.05, 2]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.linear_regressor(df_mode, y_train, df_predict, 'D')
        assert set(list_res) == set([7.982935153583617, 9.023890784982935])

    def test_knn_classifier(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8], [0.9, 1, 1], [0.95, 1.05, 1],
                                [1, 0.95, 1], [2.01, 2, 2], [2, 2, 2]], columns=['A', 'B', 'C'])
        y_train = ['a', 'b', 'a', 'a', 'a', 'b', 'b']
        df_predict = pd.DataFrame([[1.1, 0.9, 1], [2.1, 2.05, 2]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.knn_classifier(df_mode, y_train, df_predict, 'D')
        assert set(list_res) == set(['a', 'b'])

    def test_knn_regressor(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8], [0.9, 1, 1], [0.95, 1.05, 1],
                                [1, 0.95, 1], [2.01, 2, 2], [2, 2, 2]], columns=['A', 'B', 'C'])
        y_train = [8, 9, 8, 8, 8, 9, 9]
        df_predict = pd.DataFrame([[1.1, 0.9, 1], [2.1, 2.05, 2]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.knn_classifier(df_mode, y_train, df_predict, 'D')
        assert set(list_res) == set([8, 9])

    def test_svm_classifier(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8]], columns=['A', 'B', 'C'])
        y_train = ['a', 'b']
        df_predict = pd.DataFrame([[1.1, 0.9, 1], [2.1, 2.05, 2]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.svm_classifier(df_mode, y_train, df_predict, 'D')
        assert set(list_res) == set(['a', 'b'])

    def test_svm_regressor(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8]], columns=['A', 'B', 'C'])
        y_train = [8, 9]
        df_predict = pd.DataFrame([[1.1, 0.9, 1], [2.1, 2.05, 2]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.svm_regressor(df_mode, y_train, df_predict, 'D')
        assert set(list_res) == set([8.114843896546253, 8.87402432157334])

    def test_xgboost_classifier(self):
        iris = load_iris()
        x_iris = pd.DataFrame(data=np.c_[iris['data']],
                               columns=iris['feature_names'])
        x, y = iris.data, iris.target
        xtrain, xtest, ytrain, ytest = train_test_split(x_iris, y, test_size=0.15)
        list_res = self.imputer_object.xgboost_classifier(xtrain, ytrain, xtest, 'D')
        assert (len(list_res) == 23)

    def test_xgboost_regressorr(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8]], columns=['A', 'B', 'C'])
        y_train = [8, 9]
        df_predict = pd.DataFrame([[1.1, 0.9, 1], [2.1, 2.05, 2]], columns=['A', 'B', 'C'])
        list_res = self.imputer_object.xgboost_regressor(df_mode, y_train, df_predict, 'D')
        assert (7.5 < list_res[0] < 8.5) and (8.5 < list_res[1] < 9.5)

    def test_predict_with_knn_algorithm(self):
        df_mode = pd.DataFrame([[1, 1, 1.1], [2.2, 2, 1.8], [0.9, 1, 1], [0.95, 1.05, 1],
                                [1, 0.95, 1], [2.01, 2, 2], [2, 2, 2]], columns=['A', 'B', 'C'])
        df_mode2 = df_mode.copy()
        y_train = [8, 9, np.nan, 8, np.nan, 9, 9]
        target_complete = [8, 9, 8.6, 8, 8.6, 9, 9]
        df_mode['target'] = y_train
        df_mode2['target'] = target_complete
        df_res = self.imputer_object.predict(df_mode, 'knn')
        assert df_res.equals(df_mode2)