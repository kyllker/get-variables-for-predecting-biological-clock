import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.Models.supervised_models import SupervisedModel


class TestSupervisedModel:
    def setup_class(self):
        self.seed = 42
        self.supervised_model_object = SupervisedModel(self.seed, 'Target')
        self.df = pd.DataFrame([[1, 1], [1, 2]], columns=['A', 'B'])
        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])
        self.iris = load_iris()

    def test_build_an_supervised_model_object(self):
        assert isinstance(self.supervised_model_object, SupervisedModel)

    def test_linear_model(self):
        x, y = load_iris(return_X_y=True)
        df_iris = pd.DataFrame(data=np.c_[x, y],
                               columns=self.iris['feature_names'] + ['target'])

        model, best_score = self.supervised_model_object.linear_model(df_iris.iloc[:, :-1], y)
        assert isinstance(model, GridSearchCV)

    def test_xgboost_model(self):
        x, y = load_iris(return_X_y=True)
        df_iris = pd.DataFrame(data=np.c_[x, y],
                               columns=self.iris['feature_names'] + ['target'])
        model, best_score = self.supervised_model_object.xgboost_model(df_iris.iloc[:, :-1], df_iris['target'])
        assert isinstance(model, xgb.XGBRegressor)

    def test_lightgbm_model(self):
        x, y = load_iris(return_X_y=True)
        df_iris = pd.DataFrame(data=np.c_[x, y],
                               columns=self.iris['feature_names'] + ['target'])
        model = self.supervised_model_object.lightgbm_model(df_iris.iloc[:, :-1], df_iris['target'])
        assert isinstance(model, lgb.LGBMRegressor)

    def test_predict(self):
        iris = load_iris()
        x_iris = pd.DataFrame(data=np.c_[iris['data']],
                              columns=iris['feature_names'])
        x_iris['ID'] = [i for i in range(x_iris.shape[0])]
        y = iris.target
        xtrain, xtest, ytrain, ytest = train_test_split(x_iris, y, test_size=0.15)
        result, train_rmse = self.supervised_model_object.predict(xtrain, ytrain, xtest, 'ID', 42, 'Linear')
        assert (len(result[0]) == 23) and (sum([1 for i in result[1] if -1 < i < 2]))


