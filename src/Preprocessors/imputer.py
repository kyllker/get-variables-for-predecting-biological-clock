import pandas as pd
pd.options.mode.chained_assignment = None
import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import svm, preprocessing, utils
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import SGDClassifier, LinearRegression, LogisticRegression


class Imputer:

    def __init__(self, seed):
        self.seed = seed

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
    def mean_or_mode_classifier(_, y_train, x_predict):
        mode_list = max(set(y_train), key=y_train.count)
        return [mode_list for _ in range(x_predict.shape[0])]

    @staticmethod
    def mean_or_mode_regressor(_, y_train, x_predict):
        mean_list = sum(y_train) / len(y_train)
        return [mean_list for _ in range(x_predict.shape[0])]

    @staticmethod
    def linear_classifier(x_train, y_train, x_predict):
        linear_model = SGDClassifier(max_iter=1000, tol=1e-3)
        linear_model.fit(x_train, y_train)
        return linear_model.predict(x_predict)

    @staticmethod
    def linear_regressor(x_train, y_train, x_predict):
        linear_model = LinearRegression()
        linear_model.fit(x_train, y_train)
        return linear_model.predict(x_predict)

    @staticmethod
    def knn_classifier(x_train, y_train, x_predict):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)
        return knn.predict(x_predict)

    @staticmethod
    def knn_regressor(x_train, y_train, x_predict):
        if len(y_train) < 5:
            y_res = [sum(y_train) / len(y_train) for _ in range(x_predict.shape[0])]
        else:
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(x_train, y_train)
            y_res = knn.predict(x_predict)
        return y_res

    @staticmethod
    def svm_classifier(x_train, y_train, x_predict):
        clf = svm.SVC(kernel='linear')
        clf.fit(x_train, y_train)
        return clf.predict(x_predict)

    @staticmethod
    def svm_regressor(x_train, y_train, x_predict):
        clf = svm.SVR(kernel='rbf')
        clf.fit(x_train, y_train)
        return clf.predict(x_predict)

    def xgboost_classifier(self, x_train, y_train, x_predict):
        model = XGBClassifier(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8,
                              random_state=self.seed)
        model.fit(x_train, y_train)
        return model.predict(x_predict)

    def xgboost_regressor(self, x_train, y_train, x_predict):
        model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8,
                             random_state=self.seed)
        model.fit(x_train, y_train)
        return model.predict(x_predict)

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
                x_train = df_no_na_normalized.iloc[index_no_na_values, :]
                x_imput_na = df_no_na_normalized.iloc[index_na_values, :]
                y_train = [list_target_not_none[i] for i in index_no_na_values]

                if any([True if isinstance(v, str) else False for v in list_target_not_none]):
                    # Use classifier algorithm
                    if algorithm == 'mean_mode':
                        imput_res = self.mean_or_mode_classifier(x_train, y_train, x_imput_na)
                    elif algorithm == 'knn':
                        imput_res = self.knn_classifier(x_train, y_train, x_imput_na)
                    elif algorithm == 'linear':
                        imput_res = self.linear_classifier(x_train, y_train, x_imput_na)
                    elif algorithm == 'svm':
                        imput_res = self.svm_classifier(x_train, y_train, x_imput_na)
                    elif algorithm == 'xgboost':
                        imput_res = self.xgboost_classifier(x_train, y_train, x_imput_na)
                    elif algorithm == 'ensemble':
                        imput_res = self.knn_classifier(x_train, y_train, x_imput_na)

                else:
                    # Use regressor algorithm
                    if algorithm == 'mean_mode':
                        imput_res = self.mean_or_mode_regressor(x_train, y_train, x_imput_na)
                    elif algorithm == 'knn':
                        imput_res = self.knn_regressor(x_train, y_train, x_imput_na)
                    elif algorithm == 'linear':
                        imput_res = self.linear_regressor(x_train, y_train, x_imput_na)
                    elif algorithm == 'svm':
                        imput_res = self.svm_regressor(x_train, y_train, x_imput_na)
                    elif algorithm == 'xgboost':
                        imput_res = self.xgboost_regressor(x_train, y_train, x_imput_na)
                    elif algorithm == 'ensemble':
                        mean_mode_res = self.mean_or_mode_regressor(x_train, y_train, x_imput_na)
                        knn_res = self.knn_regressor(x_train, y_train, x_imput_na)
                        linear_res = self.linear_regressor(x_train, y_train, x_imput_na)
                        svm_res = self.svm_regressor(x_train, y_train, x_imput_na)
                        xgboost_res = self.xgboost_regressor(x_train, y_train, x_imput_na)
                        imput_res = [(g + h + j + k + l) / 6
                                     for g, h, j, k, l in zip(mean_mode_res, knn_res, linear_res, svm_res, xgboost_res)]

                j = 0
                for i in range(len(list_target)):
                    if i in index_na_values:
                        list_target[i] = imput_res[j]
                        j = j + 1
                dataframe.loc[:, column_na] = list_target

            return dataframe
