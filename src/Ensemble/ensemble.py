import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None
pipeline_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(pipeline_dir)

from src.Preprocessors.cleaner import Cleaner
from src.FeatureSelection.feature_selection import FeatureSelection
from src.PCA.pca import PCAModel
from src.Models.supervised_models import SupervisedModel
from src.Metrics.metrics import Metrics


class Ensemble:

    def __init__(self, seed, name_column_target):
        self.cleaner_object = Cleaner(seed)
        self.feature_selection_object = FeatureSelection(seed, name_column_target)
        self.pca_object = PCAModel(seed)
        self.supervised_model_object = SupervisedModel(seed, name_column_target)
        self.metric_object = Metrics(name_column_target)
        self.seed = seed
        self.name_column_target = name_column_target

    @staticmethod
    def split_train_test_random(dataframe, target, seed):
        x_train, x_test, y_train, y_test = \
            train_test_split(dataframe, target, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def split_train_test_manual(dataframe, target, id_column, ids_test=[0]):
        dataframe['target'] = target
        df_train = dataframe[~dataframe[id_column].isin(ids_test)]
        df_test = dataframe[dataframe[id_column].isin(ids_test)]
        y_train = df_train['target']
        y_test = df_test['target']
        x_train = df_train.drop('target', axis=1)
        x_test = df_test.drop('target', axis=1)
        return x_train, x_test, y_train, y_test

    def predict(self, df, list_columns, target, id_column, algorithm_imput='knn',
                threshold_variance=0.05, threshold_importance=30, seed=42, algorithm_supervised='Linear',
                ids_test=[], activated_pca=False, n_components_pca=20):
        print('Initial dataframe')
        print(df.shape)
        df_cleaned = self.cleaner_object.predict(df, list_columns, id_column, algorithm_imput)
        print('Cleaned dataframe')
        print(df_cleaned.shape)
        x_train, x_test, y_train, y_test = self.split_train_test_manual(df_cleaned, target, id_column, ids_test)
        print("Size X_train=" + str(x_train.shape) + " Size X_test=" + str(x_test.shape))
        # df_feature_selection, best_5_features = self.feature_selection_object.predict(
        #     df_cleaned, target, id_column, threshold_variance, threshold_importance)
        if threshold_importance == 1:
            df_feature_selection = x_train.copy()
            best_5_features = ['', '', '', '', '']
        else:
            df_feature_selection, best_5_features = self.feature_selection_object.predict(
                x_train, y_train, id_column, threshold_variance, threshold_importance)

        df_test = x_test.loc[:, df_feature_selection.columns.values.tolist()]
        print('Feature selection dataframe')
        print(df_feature_selection.shape)
        if activated_pca:
            df_pca, df_test = self.pca_object.predict(df_feature_selection, df_test, id_column,
                                                      ncomponents=n_components_pca)
        else:
            df_pca = df_feature_selection.copy()

        print('PCA dataframe')
        print(df_pca.shape)
        print('STARTING TRAINING')
        # x_train, x_test, y_train, y_test = self.split_train_test_manual(df_pca, target, id_column, ids_test)
        # print("Size X_train="+str(x_train.shape) + " Size X_test="+str(x_test.shape))
        list_supervided_result, train_rmse = self.supervised_model_object.predict(df_pca,
                                                                                  y_train,
                                                                                  df_test, id_column,
                                                                                  seed, algorithm_supervised)
        list_result = [list_supervided_result[0], list_supervided_result[1], list(y_test)]
        rmse, mae, r2, df_results = self.metric_object.predict(list_result)
        return rmse, mae, r2, df_results, best_5_features, train_rmse
