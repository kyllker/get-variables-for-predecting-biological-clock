import os
import sys
import re
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
project_dir = os.path.join(os.path.dirname(__file__), '..', 'detect_fitbit_features')
sys.path.append(project_dir)


class FeatureSelection:

    def __init__(self, seed):
        self.seed = seed

    @staticmethod
    def drop_columns_little_variance(dataframe, threshold_variance):
        selector = VarianceThreshold(threshold_variance)
        selector.fit(dataframe)
        df_columns_big_variance = dataframe[dataframe.columns[selector.get_support(indices=True)]]
        return df_columns_big_variance

    def feature_selection_with_ml_algorithms(self, dataframe, target, threshold_importance):
        df_aux = dataframe.copy()
        x_model, x_valid, y_model, y_valid = train_test_split(df_aux, target, random_state=self.seed, test_size=0.2)
        model_dict = {
            'SVM': SVR(kernel='linear'),
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(random_state=self.seed, n_jobs=-1),
            'XGBoostRegressor': XGBRegressor(objective='reg:squarederror', random_state=self.seed)
        }
        estimator_dict = {}
        importance_features_sorted_all = pd.DataFrame()
        for model_name, model in model_dict.items():
            model.fit(x_model, y_model)
            if (model_name == 'LinearRegression') or (model_name == 'SVM'):
                importance_values = np.absolute(model.coef_)
            else:
                importance_values = model.feature_importances_
            importance_features_sorted = pd.DataFrame(
                importance_values.reshape([-1, len(df_aux.columns)]),
                columns=df_aux.columns).mean(axis=0).sort_values(ascending=False).to_frame()
            importance_features_sorted.rename(columns={0: 'feature_importance'}, inplace=True)
            importance_features_sorted['ranking'] = importance_features_sorted['feature_importance'].rank(ascending=False)
            importance_features_sorted['model'] = model_name
            importance_features_sorted_all = importance_features_sorted_all.append(importance_features_sorted)
            estimator_dict[model_name] = model
            importance_features_sorted_all['feature'] = importance_features_sorted_all.index
        importance_features_sorted_all = importance_features_sorted_all.reset_index(drop=True)
        df_important_threshold = \
            importance_features_sorted_all[importance_features_sorted_all['feature_importance'] > threshold_importance]
        selection_features = list(df_important_threshold.feature.unique())

        return dataframe.loc[:, selection_features]

    def predict(self, dataframe, target, threshold_variance=0.05, threshold_importance=0.3):
        dataframe_big_variance = self.drop_columns_little_variance(dataframe, threshold_variance)
        dataframe_ml_selection = \
            self.feature_selection_with_ml_algorithms(dataframe_big_variance, target, threshold_importance)
        return dataframe_ml_selection
