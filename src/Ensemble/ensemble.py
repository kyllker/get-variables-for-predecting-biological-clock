import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None
pipeline_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(pipeline_dir)

from src.Preprocessors.cleaner import Cleaner
from src.FeatureSelection.feature_selection import FeatureSelection
from src.Models.supervised_models import SupervisedModel


class Ensemble:

    def __init__(self):
        self.cleaner_object = Cleaner()
        self.feature_selection_object = FeatureSelection()
        self.supervised_model_object = SupervisedModel()

    @staticmethod
    def read_data(filename, sheet):
        if 'tests' in os.getcwd():
            xl_file = pd.ExcelFile(os.path.join('..', 'src', 'Data', filename))
        else:
            xl_file = pd.ExcelFile(os.path.join('src', 'Data', filename))

        dfs = {sheet_name: xl_file.parse(sheet_name)
               for sheet_name in xl_file.sheet_names}

        return dfs[sheet].iloc[:60, :]

    @staticmethod
    def split_train_test(dataframe, target, seed):
        X_train, X_test, y_train, y_test = \
            train_test_split(dataframe, target, test_size=0.2, random_state=seed)
        return X_train, X_test, y_train, y_test

    def predict(self, filename, sheet, list_columns, target, algorithm_imput='knn',
                threshold_variance=0.05, threshold_importance=0.3, seed=42, algorithm_supervised='Linear'):
        df = self.read_data(filename, sheet)
        print('initial dataframe')
        print(df.shape)
        df_cleaned = self.cleaner_object.predict(df, list_columns, algorithm_imput)
        print('df_cleaned')
        print(df_cleaned.shape)
        df_feature_selection = self.feature_selection_object.predict(
            df_cleaned, target, threshold_variance, threshold_importance)
        print('df_feature_selection')
        print(df_feature_selection.shape)
        X_train, X_test, y_train, y_test = self.split_train_test(df_feature_selection, target, seed)
        print("size X_train="+str(X_train.shape) + " size X_test="+str(X_test.shape))
        print("len y_train=" + str(len(y_train)) + " len y_test=" + str(len(y_test)))
        df_supervided_result = self.supervised_model_object.predict(X_train, y_train, X_test,
                                                                    seed, algorithm_supervised)
        print('df_supervided_result')
        print(df_supervided_result.shape)
        return [df_supervided_result, y_test]
