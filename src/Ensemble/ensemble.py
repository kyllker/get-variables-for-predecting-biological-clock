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

    def __init__(self, seed):
        self.cleaner_object = Cleaner(seed)
        self.feature_selection_object = FeatureSelection(seed)
        self.pca_object = PCAModel(seed)
        self.supervised_model_object = SupervisedModel(seed)
        self.metric_object = Metrics()
        self.seed = seed

    @staticmethod
    def read_data(filename, sheet):
        if 'tests' in os.getcwd():
            xl_file = pd.ExcelFile(os.path.join('..', 'Data', filename))
        else:
            xl_file = pd.ExcelFile(os.path.join('Data', filename))

        dfs = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}

        return dfs[sheet].iloc[:60, :]

    @staticmethod
    def split_train_test_random(dataframe, target, seed):
        x_train, x_test, y_train, y_test = \
            train_test_split(dataframe, target, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def split_train_test_manual(dataframe, target, ids_test=[0]):
        dataframe['target'] = target
        df_train = dataframe[~dataframe['ID_Muestra'].isin(ids_test)]
        df_test = dataframe[dataframe['ID_Muestra'].isin(ids_test)]
        y_train = df_train['target']
        y_test = df_test['target']
        x_train = df_train.drop('target', axis=1)
        x_test = df_test.drop('target', axis=1)
        return x_train, x_test, y_train, y_test

    def predict(self, filename, sheet, list_columns, target, algorithm_imput='knn',
                threshold_variance=0.05, threshold_importance=0.3, seed=42, algorithm_supervised='Linear',
                ids_test=[0], activated_pca=False, n_components_pca=20):
        df = self.read_data(filename, sheet)
        print('Initial dataframe')
        print(df.shape)
        if False:
            df_cleaned = pd.read_csv('/home/kyllker/Desktop/TFM/get-variables-for-predecting-biological-clock/Data/df_cleaned.csv')
        else:
            df_cleaned = self.cleaner_object.predict(df, list_columns, algorithm_imput)
        print('Cleaned dataframe')
        print(df_cleaned.shape)
        df_feature_selection = self.feature_selection_object.predict(
            df_cleaned, target, threshold_variance, threshold_importance)
        print('Feature selection dataframe')
        print(df_feature_selection.shape)
        if activated_pca:
            df_pca = self.pca_object.predict(df_feature_selection, ncomponents=n_components_pca)
        else:
            df_pca = df_feature_selection.copy()
        print('PCA dataframe')
        print(df_pca.shape)
        # x_train, x_test, y_train, y_test = self.split_train_test_random(df_pca, target, seed)
        x_train, x_test, y_train, y_test = self.split_train_test_manual(df_pca, target, ids_test)
        print("Size X_train="+str(x_train.shape) + " Size X_test="+str(x_test.shape))
        list_supervided_result = self.supervised_model_object.predict(x_train, y_train, x_test,
                                                                      seed, algorithm_supervised)
        list_result = [list_supervided_result[0], list_supervided_result[1], list(y_test)]
        return self.metric_object.predict(list_result, algorithm_supervised)
