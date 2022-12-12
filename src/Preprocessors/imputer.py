import pandas as pd
pd.options.mode.chained_assignment = None
import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import svm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import SGDClassifier, LinearRegression


class Imputer:

    def __init__(self):
        pass

    @staticmethod
    def check_there_are_na_values(dataframe):
        if dataframe.isna().sum().sum() > 0:
            return True
        else:
            return False

    @staticmethod
    def get_na_and_no_na_columns(dataframe):
        dataframe_sum_na = dataframe.isna().sum()
        dataframe_nas_by_column = dataframe_sum_na.loc[lambda x: x > 0]
        list_na_columns = list(dataframe_nas_by_column.sort_values(ascending=True).index)
        list_no_na_columns = list(set(list(dataframe.columns)) - set(list_na_columns))
        df_no_na = dataframe[list_no_na_columns]
        column_numeric = df_no_na.columns[list(map(pd.api.types.is_numeric_dtype, df_no_na.dtypes))]
        df_no_na_numeric = df_no_na[column_numeric]
        list_no_na_columns_numeric = list(df_no_na_numeric.columns)
        return [list_na_columns, list_no_na_columns_numeric]

    @staticmethod
    def normalize_dataframe(dataframe):
        normalized_dataframe = dataframe.copy()
        for index_column in range(normalized_dataframe.shape[1]):
            column_name = normalized_dataframe.columns[index_column]
            if normalized_dataframe[column_name].nunique() == 1:
                normalized_dataframe.loc[:, column_name] = 1
            else:
                normalized_dataframe.loc[:, column_name] = (normalized_dataframe[column_name] - normalized_dataframe[column_name].min()) / (
                    normalized_dataframe[column_name].max() - normalized_dataframe[column_name].min())
        return normalized_dataframe

    @staticmethod
    def linear_classifier(X_train, y_train, X_predict):
        linear_model = SGDClassifier(max_iter=1000, tol=1e-3)
        linear_model.fit(X_train, y_train)
        return linear_model.predict(X_predict)

    @staticmethod
    def linear_regressor(X_train, y_train, X_predict):
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        return linear_model.predict(X_predict)

    @staticmethod
    def knn_classifier(X_train, y_train, X_predict):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        return knn.predict(X_predict)

    @staticmethod
    def knn_regressor(X_train, y_train, X_predict):
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        return knn.predict(X_predict)

    @staticmethod
    def svm_classifier(X_train, y_train, X_predict):
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        return clf.predict(X_predict)

    @staticmethod
    def svm_regressor(X_train, y_train, X_predict):
        clf = svm.SVR(kernel='rbf')
        clf.fit(X_train, y_train)
        return clf.predict(X_predict)

    @staticmethod
    def xgboost_classifier(X_train, y_train, X_predict):
        model = XGBClassifier(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        model.fit(X_train, y_train)
        return model.predict(X_predict)

    @staticmethod
    def xgboost_regressor(X_train, y_train, X_predict):
        model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        model.fit(X_train, y_train)
        return model.predict(X_predict)

    def predict(self, dataframe, algorithm):
        if not self.check_there_are_na_values(dataframe):
            return dataframe
        else:
            list_na_columns = self.get_na_and_no_na_columns(dataframe)[0]
            for column_na in list_na_columns:
                list_no_na_columns = self.get_na_and_no_na_columns(dataframe)[1]
                df_no_na = dataframe[list_no_na_columns]
                df_no_na_normalized = self.normalize_dataframe(df_no_na)
                list_target = list(dataframe[column_na])
                list_target_not_none = [np.nan if v is None else v for v in list_target]
                index_na_values = [i for i, x in enumerate(list_target) if pd.isnull(x)]
                index_no_na_values = list(set([i for i in range(len(list_target))]) - set(index_na_values))
                X_train = df_no_na_normalized.iloc[index_no_na_values, :]
                X_imput_na = df_no_na_normalized.iloc[index_na_values, :]
                y_train = [list_target_not_none[i] for i in index_no_na_values]

                if any([True if isinstance(v, str) else False for v in list_target_not_none]):
                    # Use classifier algorithm
                    if algorithm == 'knn':
                        input_res = self.knn_classifier(X_train, y_train, X_imput_na)
                    elif algorithm == 'linear':
                        input_res = self.linear_classifier(X_train, y_train, X_imput_na)
                    elif algorithm == 'svm':
                        input_res = self.svm_classifier(X_train, y_train, X_imput_na)
                    elif algorithm == 'xgboost':
                        input_res = self.xgboost_classifier(X_train, y_train, X_imput_na)
                else:
                    # Use regressor algorithm
                    if algorithm == 'knn':
                        input_res = self.knn_regressor(X_train, y_train, X_imput_na)
                    elif algorithm == 'linear':
                        input_res = self.linear_regressor(X_train, y_train, X_imput_na)
                    elif algorithm == 'svm':
                        input_res = self.svm_regressor(X_train, y_train, X_imput_na)
                    elif algorithm == 'xgboost':
                        input_res = self.xgboost_regressor(X_train, y_train, X_imput_na)

                j = 0
                for i in range(len(list_target)):
                    if i in index_na_values:
                        list_target[i] = input_res[j]
                        j = j + 1
                dataframe.loc[:, column_na] = list_target

            return dataframe
