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
from src.Metrics.metrics import Metrics


class Ensemble:

    def __init__(self, seed):
        self.cleaner_object = Cleaner(seed)
        self.feature_selection_object = FeatureSelection(seed)
        self.supervised_model_object = SupervisedModel(seed)
        self.result_object = Metrics()

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
    def split_train_test_random(dataframe, target, seed):
        x_train, x_test, y_train, y_test = \
            train_test_split(dataframe, target, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def split_train_test_manual(dataframe, target, seed):
        x_train, x_test, y_train, y_test = \
            train_test_split(dataframe, target, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test

    def predict(self, filename, sheet, list_columns, target, algorithm_imput='knn',
                threshold_variance=0.05, threshold_importance=0.3, seed=42, algorithm_supervised='Linear'):
        df = self.read_data(filename, sheet)
        print('Initial dataframe')
        print(df.shape)
        df_cleaned = self.cleaner_object.predict(df, list_columns, algorithm_imput)
        print('Cleaned dataframe')
        print(df_cleaned.shape)
        df_feature_selection = self.feature_selection_object.predict(
            df_cleaned, target, threshold_variance, threshold_importance)
        print('Feature selection dataframe')
        print(df_feature_selection.shape)
        x_train, x_test, y_train, y_test = self.split_train_test_random(df_feature_selection, target, seed)
        print("Size X_train="+str(x_train.shape) + " Size X_test="+str(x_test.shape))
        df_supervided_result = self.supervised_model_object.predict(x_train, y_train, x_test,
                                                                    seed, algorithm_supervised)
        list_result = [df_supervided_result, y_test]
        return self.result_object.predict(list_result)
