import pandas as pd
pd.options.mode.chained_assignment = None
import os
import sys
import numbers
import numpy as np
project_dir = os.path.join(os.path.dirname(__file__), '..', 'detect_fitbit_features')
sys.path.append(project_dir)
from src.Preprocessors.imputer import Imputer


class Cleaner:

    def __init__(self, seed):
        self.imputer = Imputer(seed)

    @staticmethod
    def filter_desired_columns(dataframe, list_columns_with_order):
        try:
            return dataframe.iloc[:, list_columns_with_order]
        except:
            return dataframe

    @staticmethod
    def remove_duplicate_columns(dataframe):
        return dataframe.loc[:, ~dataframe.apply(lambda x: x.duplicated(), axis=1).all()].copy()

    @staticmethod
    def remove_constant_columns(dataframe):
        for column in dataframe.columns:
            if len(dataframe[column].unique()) == 1:
                dataframe = dataframe.drop(columns=[column])
        return dataframe

    @staticmethod
    def convert_to_numerical_values_column_with_two_different_values(dataframe):
        df_aux = dataframe.copy()
        for index_column in range(df_aux.shape[1]):
            column_name = df_aux.columns[index_column]
            list_unique_values = dataframe[column_name].unique()
            if len(list_unique_values) == 2:
                if not all([isinstance(i, numbers.Number) for i in list_unique_values]):
                    new_column = pd.DataFrame(dataframe.loc[:, column_name])
                    new_column_name = column_name + '_' + str(list_unique_values[0]) + '1_' +str(list_unique_values[1]) + '0'
                    new_column = new_column.rename(columns={column_name: new_column_name})
                    dataframe = pd.concat([dataframe, new_column], 1)
                    if list_unique_values[0] == list_unique_values[0]:
                        dataframe.loc[dataframe[new_column_name] == list_unique_values[0], new_column_name] = 0
                    else:
                        dataframe.loc[dataframe[new_column_name].isna(), new_column_name] = 0
                    if list_unique_values[1] == list_unique_values[1]:
                        dataframe.loc[dataframe[new_column_name] == list_unique_values[1], new_column_name] = 1
                    else:
                        dataframe.loc[dataframe[new_column_name].isna(), new_column_name] = 1

                    dataframe = dataframe.drop(columns=[column_name])
        return dataframe

    @staticmethod
    def convert_to_numerical_values_column_with_more_than_two_different_values(dataframe):
        df_with_dummies = dataframe.copy()
        list_columns_non_numerical_values = []
        for index_column in range(dataframe.shape[1]):
            column_name = dataframe.columns[index_column]
            list_unique_values = dataframe[column_name].unique()
            if len(list_unique_values) > 2:
                if not all([isinstance(i, numbers.Number) for i in list(dataframe.iloc[:, index_column])]):
                    list_columns_non_numerical_values.append(index_column)
        for index_column in list_columns_non_numerical_values:
            list_unique_values = dataframe[dataframe.columns[index_column]].unique()
            if 2 < len(list_unique_values) <= 5:
                column_name = dataframe.columns[index_column]
                for value in list_unique_values:
                    new_column = pd.DataFrame(dataframe.loc[:, column_name])
                    if value == value:
                        new_column = new_column.rename(columns={column_name: column_name + '_' + str(value)})
                        df_with_dummies = pd.concat([df_with_dummies, new_column], 1)
                        df_with_dummies.loc[
                            df_with_dummies[column_name + '_' + str(value)] != value, column_name + '_' + str(
                                value)] = 0
                        df_with_dummies.loc[
                            df_with_dummies[column_name + '_' + str(value)] == value, column_name + '_' + str(
                                value)] = 1
                    else:
                        new_column = new_column.rename(columns={column_name: column_name + '_nan'})
                        df_with_dummies = pd.concat([df_with_dummies, new_column], 1)
                        df_with_dummies.loc[
                            df_with_dummies[df_with_dummies.columns[index_column]].isna(), column_name + '_nan'] = 1
                        df_with_dummies.loc[df_with_dummies[df_with_dummies.columns[
                            index_column]].isna() == False, column_name + '_nan'] = 0
        for index_column in list_columns_non_numerical_values:
            column_name = dataframe.columns[index_column]
            df_with_dummies = df_with_dummies.drop(columns=[column_name])

        return df_with_dummies

    @staticmethod
    def normalize_dataframe(dataframe):
        normalized_dataframe = dataframe.copy()
        for index_column in range(normalized_dataframe.shape[1]):
            column_name = normalized_dataframe.columns[index_column]
            if normalized_dataframe[column_name].nunique() == 1:
                normalized_dataframe.loc[:, column_name] = 1
            else:
                normalized_dataframe.loc[:, column_name] = \
                    (normalized_dataframe[column_name] - normalized_dataframe[column_name].min()) / \
                    (normalized_dataframe[column_name].max() - normalized_dataframe[column_name].min())
        return normalized_dataframe

    def predict(self, dataframe, list_columns_with_order, algorithm='knn'):
        dataframe_desired_columns = self.filter_desired_columns(dataframe, list_columns_with_order)
        dataframe_numerical_values = \
            self.convert_to_numerical_values_column_with_two_different_values(dataframe_desired_columns)
        dataframe_no_na = self.imputer.predict(dataframe_numerical_values, algorithm)
        dataframe_numerical_values = \
            self.convert_to_numerical_values_column_with_two_different_values(dataframe_no_na)
        dataframe_only_numerical_values = \
            self.convert_to_numerical_values_column_with_more_than_two_different_values(dataframe_numerical_values)
        dataframe_remove_duplicate_columns = self.remove_duplicate_columns(dataframe_only_numerical_values)
        dataframe_remove_constant_columns = self.remove_constant_columns(dataframe_remove_duplicate_columns)

        dataframe_normalized = self.normalize_dataframe(dataframe_remove_constant_columns)

        return dataframe_normalized
