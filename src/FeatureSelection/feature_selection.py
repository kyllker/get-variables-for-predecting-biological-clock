import pickle
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class FeatureSelection:

    def __init__(self, seed, name_column_target):
        self.seed = seed
        self.name_column_target = name_column_target

    @staticmethod
    def drop_columns_little_variance(dataframe, threshold_variance):
        selector = VarianceThreshold(threshold_variance)
        selector.fit(dataframe)
        df_columns_big_variance = dataframe[dataframe.columns[selector.get_support(indices=True)]]
        return df_columns_big_variance

    def feature_selection_with_ml_algorithms(self, dataframe, target, threshold_importance):
        # Probar los modelos por separado y con el ensemble dejarlo como está aquí
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
            x_model = x_model.astype(float)
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
            importance_features_sorted_all[importance_features_sorted_all['ranking'] < threshold_importance]
        selection_features = list(df_important_threshold.feature.unique())

        return dataframe.loc[:, selection_features]

    def predict(self, dataframe, target, id_column, threshold_variance=0.05, threshold_importance=0.3):
        id_muestra = pd.DataFrame(dataframe[id_column])
        dataframe_no_id = dataframe.drop(id_column, axis=1)
        dataframe_big_variance = self.drop_columns_little_variance(dataframe_no_id, threshold_variance)
        dataframe_ml_selection = \
            self.feature_selection_with_ml_algorithms(dataframe_big_variance, target, threshold_importance)
        with open(os.path.join('model_store', 'saved_models', 'feature_selection',
                               self.name_column_target + '_columns_selected_' +
                               str(threshold_variance).replace('.', '') + '_'
                               + str(threshold_importance).replace('.', '') + '.pkl'), 'wb') as f:
            pickle.dump(dataframe_ml_selection.columns.values.tolist(), f)
        cleaned_dataframe = pd.concat([id_muestra, dataframe_ml_selection], 1)
        return cleaned_dataframe
