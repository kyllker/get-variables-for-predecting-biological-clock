import pandas as pd
import numbers
import pickle
import os
from src.Preprocessors.imputer import Imputer

pd.options.mode.chained_assignment = None


class Cleaner:

    def __init__(self, seed):
        self.imputer = Imputer(seed)

    @staticmethod
    def filter_desired_columns(dataframe, list_columns_with_name, id_column):
        if id_column in list_columns_with_name:
            list_columns_with_name.remove(id_column)
        try:
            with open(
                os.path.join('model_store', 'saved_models', 'cleaner', 'columns_before_imput.pkl'),
                    'wb') as f:
                pickle.dump(list_columns_with_name, f)
            return dataframe.loc[:, list_columns_with_name]
        except:
            print(os.getcwd())
            print(os.listdir(os.getcwd()))
            with open(os.path.join('model_store', 'saved_models', 'cleaner', 'columns_before_imput.pkl'),
                      'wb') as f:
                pickle.dump(dataframe.columns.values.tolist(), f)
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
        columns_two_different_values = []
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
                        columns_two_different_values.append((new_column_name, list_unique_values[0], 0))
                    else:
                        dataframe.loc[dataframe[new_column_name].isna(), new_column_name] = 0
                        columns_two_different_values.append((new_column_name, "nan", 0))
                    if list_unique_values[1] == list_unique_values[1]:
                        dataframe.loc[dataframe[new_column_name] == list_unique_values[1], new_column_name] = 1
                        columns_two_different_values.append((new_column_name, list_unique_values[0], 1))
                    else:
                        columns_two_different_values.append((new_column_name, "nan", 1))
                        dataframe.loc[dataframe[new_column_name].isna(), new_column_name] = 1

                    dataframe = dataframe.drop(columns=[column_name])
        if len(columns_two_different_values) > 0:
            with open(
                os.path.join('model_store', 'saved_models', 'cleaner', 'columns_two_different_values.pkl'),
                    'wb') as f:
                pickle.dump(columns_two_different_values, f)
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
            list_unique_values = list(dataframe[dataframe.columns[index_column]].unique())
            list_column_dummy = [dataframe.columns[index_column], list_unique_values]

            with open(
                    os.path.join('model_store', 'saved_models', 'cleaner', 'dummies',
                                 'dummy_column_' + dataframe.columns[index_column] + '.pkl'),
                    'wb') as f:
                pickle.dump(list_column_dummy, f)
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
                        df_with_dummies[column_name + '_' + str(value)] = \
                            df_with_dummies[column_name + '_' + str(value)].astype(int)
                    else:
                        new_column = new_column.rename(columns={column_name: column_name + '_nan'})
                        df_with_dummies = pd.concat([df_with_dummies, new_column], 1)
                        df_with_dummies.loc[
                            df_with_dummies[df_with_dummies.columns[index_column]].isna(), column_name + '_nan'] = 1
                        df_with_dummies.loc[df_with_dummies[df_with_dummies.columns[
                            index_column]].isna() == False, column_name + '_nan'] = 0
                        df_with_dummies[column_name + '_nan'] = \
                            df_with_dummies[column_name + '_nan'].astype(int)
        for index_column in list_columns_non_numerical_values:
            column_name = dataframe.columns[index_column]
            df_with_dummies = df_with_dummies.drop(columns=[column_name])

        return df_with_dummies

    @staticmethod
    def normalize_dataframe(dataframe):
        list_columns_normalized = []
        normalized_dataframe = dataframe.copy()
        for index_column in range(normalized_dataframe.shape[1]):
            column_name = normalized_dataframe.columns[index_column]
            if normalized_dataframe[column_name].nunique() == 1:
                normalized_dataframe.loc[:, column_name] = 1
                list_columns_normalized.append((column_name, 1, 1))
            else:
                list_columns_normalized.append((column_name, normalized_dataframe[column_name].min(),
                                                (normalized_dataframe[column_name].max())))
                normalized_dataframe.loc[:, column_name] = \
                    (normalized_dataframe[column_name] - normalized_dataframe[column_name].min()) / \
                    (normalized_dataframe[column_name].max() - normalized_dataframe[column_name].min())
        with open(os.path.join('model_store', 'saved_models', 'cleaner', 'normalize_columns.pkl'),
                  'wb') as f:
            pickle.dump(list_columns_normalized, f)
        return normalized_dataframe

    def predict(self, dataframe, list_columns_with_names, id_column, algorithm='knn'):
        id_muestra = pd.DataFrame(dataframe[id_column])
        dataframe_no_id = dataframe.drop(id_column, axis=1)
        dataframe_desired_columns = self.filter_desired_columns(dataframe_no_id, list_columns_with_names, id_column)
        dataframe_numerical_values = \
            self.convert_to_numerical_values_column_with_two_different_values(dataframe_desired_columns)

        dataframe_only_numerical_values = \
            self.convert_to_numerical_values_column_with_more_than_two_different_values(dataframe_numerical_values)
        dataframe_remove_duplicate_columns = self.remove_duplicate_columns(dataframe_only_numerical_values)
        dataframe_remove_constant_columns = self.remove_constant_columns(dataframe_remove_duplicate_columns)

        dataframe_normalized = self.normalize_dataframe(dataframe_remove_constant_columns)

        dataframe_no_na = self.imputer.predict(dataframe_normalized, algorithm)

        cleaned_dataframe = pd.concat([id_muestra, dataframe_no_na], 1)
        return cleaned_dataframe
