import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from unittest.mock import patch
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.Ensemble.ensemble import Ensemble


class TestEnsemble:
    def setup_class(self):
        self.seed = 42
        self.ensemble_object = Ensemble(self.seed, 'ID_Muestra')
        self.df = pd.DataFrame([[1, 1], [1, 2]], columns=['A', 'B'])
        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])
        self.prefix_path = 'src.'

    def test_build_an_ensemble_object(self):
        assert isinstance(self.ensemble_object, Ensemble)

    def test_split_train_test_random(self):
        iris = load_iris()
        x_iris = pd.DataFrame(data=np.c_[iris['data']],
                              columns=iris['feature_names'])
        x_iris['ID'] = [i for i in range(x_iris.shape[0])]
        y = iris.target
        x_train, x_test, y_train, y_test = self.ensemble_object.split_train_test_random(x_iris, y, 42)
        assert (x_train.shape[0] == 120) and (x_train.shape[1] == 5)

    def test_split_train_test_manual(self):
        iris = load_iris()
        x_iris = pd.DataFrame(data=np.c_[iris['data']],
                              columns=iris['feature_names'])
        x_iris['ID'] = [i for i in range(x_iris.shape[0])]
        y = iris.target
        x_train, x_test, y_train, y_test = self.ensemble_object.split_train_test_manual(x_iris, y, 'ID', [1, 2, 3])
        assert (x_train.shape[0] == 147) and (x_train.shape[1] == 5) and (set(list(x_test['ID'])) == set([1, 2, 3]))

    def test_predict_calls_all_(self,
                                _cleaner_predict_mock,
                                _feature_selection_predict_mock,
                                _pca_model_predict_mock,
                                _supervised_models_predict_mock,
                                _metrics_predict_mock):
        iris = load_iris()
        df_iris = pd.DataFrame(data=np.c_[iris['data']],
                              columns=iris['feature_names'])
        df_iris['ID'] = [i for i in range(df_iris.shape[0])]
        df_iris['Target'] = iris.target
        list_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                        'petal width (cm)', 'ID']
        res_ensemble = self.ensemble_object.predict(df_iris, list_columns, list(df_iris['Target']), 'ID', 'knn', 0.01,
                                                    30, 42, 'Linear', [1, 2, 3], True, 10)

        assert (res_ensemble[0] == 1) and (res_ensemble[1] == 1) and (res_ensemble[2] == 1)

    @pytest.fixture()
    def _cleaner_predict_mock(self, mocker):
        path = 'Preprocessors.cleaner.Cleaner'
        iris = load_iris()
        df_iris = pd.DataFrame(data=np.c_[iris['data']],
                               columns=iris['feature_names'])
        df_iris['ID'] = [i for i in range(df_iris.shape[0])]
        with mocker.patch(self.prefix_path + path + '.predict', return_value=df_iris):
            result = self.ensemble_object
            yield result

    @pytest.fixture()
    def _feature_selection_predict_mock(self, mocker):
        iris = load_iris()
        df_iris = pd.DataFrame(data=np.c_[iris['data']],
                               columns=iris['feature_names'])
        df_iris['ID'] = [i for i in range(df_iris.shape[0])]
        path = 'FeatureSelection.feature_selection.FeatureSelection'
        with mocker.patch(
                self.prefix_path + path + '.predict',
                return_value=[df_iris,
                              ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]):
            result = self.ensemble_object
            yield result

    @pytest.fixture()
    def _pca_model_predict_mock(self, mocker):
        path = 'PCA.pca.PCAModel'
        iris = load_iris()
        df_iris = pd.DataFrame(data=np.c_[iris['data']],
                               columns=iris['feature_names'])
        df_iris['ID'] = [i for i in range(df_iris.shape[0])]
        with mocker.patch(self.prefix_path + path + '.predict', return_value=df_iris):
            result = self.ensemble_object
            yield result

    @pytest.fixture()
    def _supervised_models_predict_mock(self, mocker):
        path = 'Models.supervised_models.SupervisedModel'
        with mocker.patch(self.prefix_path + path + '.predict', return_value=[[1, 2, 3], [0, 1, 0]]):
            result = self.ensemble_object
            yield result

    @pytest.fixture()
    def _metrics_predict_mock(self, mocker):
        path = 'Metrics.metrics.Metrics'
        df = pd.DataFrame([[1, 1], [1.1, 0.95], [1, 0.99]], columns=['a', 'b'])
        with mocker.patch(self.prefix_path + path + '.predict', return_value=[1, 1, 1, df]):
            result = self.ensemble_object
            yield result