import zipfile
import pandas as pd
import pickle
import os
import shutil


class Predictor:
    def __init__(self):
        pass

    @staticmethod
    def load_pickle(path_pickle):
        file = open(path_pickle, 'rb')
        pickle_file = pickle.load(file)
        file.close()
        return pickle_file

    def desired_columns_before_all(self, dataframe, path_best_model):
        list_columns_desired = \
            self.load_pickle(os.path.join(path_best_model, "cleaner", "columns_before_imput.pkl"))
        return dataframe.loc[:, list_columns_desired]

    @staticmethod
    def remove_duplicate_columns(dataframe):
        return dataframe.loc[:, ~dataframe.apply(lambda x: x.duplicated(), axis=1).all()].copy()

    @staticmethod
    def remove_constant_columns(dataframe):
        for column in dataframe.columns:
            if len(dataframe[column].unique()) == 1:
                dataframe = dataframe.drop(columns=[column])
        return dataframe

    def convert_dummies(self, dataframe, path_best_model):
        dataframe = dataframe.reset_index(drop=True)
        path_dummies = os.listdir(os.path.join(path_best_model, "cleaner", "dummies"))
        for col_dummy_path in range(len(path_dummies)):
            dummy_column_values = \
                self.load_pickle(os.path.join(path_best_model, "cleaner", "dummies", path_dummies[col_dummy_path]))
            col_dummy = dummy_column_values[0]
            if col_dummy in dataframe.columns.values.tolist():
                for value_dummy in dummy_column_values[1]:
                    dataframe[col_dummy + '_' + value_dummy] = 0
                    for i in range(dataframe.shape[0]):
                        if dataframe.loc[i, col_dummy] == value_dummy:
                            dataframe.loc[i, col_dummy + '_' + value_dummy] = 1
                dataframe = dataframe.drop(col_dummy, axis=1)
        return dataframe

    def normalized_columns(self, dataframe, path_best_model):
        dataframe = dataframe.reset_index(drop=True)
        list_columns_normalized = \
            self.load_pickle(os.path.join(path_best_model, "cleaner", "normalize_columns.pkl"))
        names_columns_in_list_columns_normalized = [tup[0] for tup in list_columns_normalized]
        for column in dataframe.columns:
            if column in names_columns_in_list_columns_normalized:
                for i in range(dataframe.shape[0]):
                    if not pd.isnull(dataframe.loc[i, column]):
                        index_column = names_columns_in_list_columns_normalized.index(column)
                        min_value = list_columns_normalized[index_column][1]
                        max_value = list_columns_normalized[index_column][2]
                        if min_value == 1:
                            dataframe.loc[i, column] = 1
                        else:
                            dataframe.loc[i, column] = (dataframe.loc[i, column] - min_value) / (max_value - min_value)
        return dataframe

    def imputer_predict(self, dataframe, path_best_model, algorithm_imput):
        dataframe = dataframe.reset_index(drop=True)

        list_columns_to_imputer_sorted = self.load_pickle(
            os.path.join(path_best_model, "imputer", "order_columns_imputer.pkl"))
        list_columns_to_imputer_sorted = [col for col in list_columns_to_imputer_sorted
                                          if col in dataframe.columns.values.tolist()]

        for column in list_columns_to_imputer_sorted:
            if dataframe[column].isnull().values.any():
                model_and_columns = \
                    self.load_pickle(
                        os.path.join(path_best_model, 'imputer', algorithm_imput + '_regressor_' + column + '.pkl'))
                listcolumns = model_and_columns[1]
                list_cols_dataframe = dataframe.columns.values.tolist()
                try:
                    df_aux = dataframe.loc[:, model_and_columns[1]].reset_index(drop=True)
                except:
                    listcolumns = [col for col in listcolumns if col in list_cols_dataframe]
                    df_aux = dataframe.loc[:, listcolumns].reset_index(drop=True)

                if df_aux.isna().sum().sum() > 0:
                    for col in df_aux.columns:
                        if df_aux[col].isna().sum().sum() > 0:
                            model_and_columns_mean = \
                                self.load_pickle(
                                    os.path.join(path_best_model, 'imputer', 'mean_regressor_' + col + '.pkl'))
                            for row_df_aux in range(df_aux.shape[0]):
                                if str(df_aux.loc[row_df_aux, col]) == 'nan':
                                    df_aux.loc[row_df_aux, col] = model_and_columns_mean[1]

                for i in range(dataframe.shape[0]):
                    if str(dataframe.loc[i, column]) == 'nan':
                        if (algorithm_imput == 'mean') or (algorithm_imput == 'mode'):
                            dataframe.loc[i, column] = model_and_columns[1]
                        else:
                            dataframe.loc[i, column] = model_and_columns[0].predict(pd.DataFrame(df_aux.iloc[i, :]).T)
        return dataframe

    def feature_selection_predict(self, dataframe, path_best_model, best_parameters):
        list_columns_selection = \
            self.load_pickle(os.path.join(path_best_model, "feature_selection", "columns_selected_" +
                                          str(best_parameters.get('threshold_variance')).replace('.', '') + '_' +
                                          str(best_parameters.get('threshold_importance')).replace('.', '') + ".pkl"))

        return dataframe.loc[:, list_columns_selection]

    def pca_predict(self, dataframe, path_best_model, best_parameters):
        pca_model = \
            self.load_pickle(os.path.join(path_best_model, "pca", "pca_model_" +
                                          str(best_parameters.get('n_components_pca')) + ".pkl"))
        edad_cronologica = pd.DataFrame(dataframe['Edad_Cronologica'])
        dataframe_no_edad = dataframe.loc[:, pca_model[1]]
        df_pca = pd.DataFrame(pca_model[0].transform(dataframe_no_edad))
        df_reduced_dimensionality = pd.concat([edad_cronologica, df_pca], 1)
        return df_reduced_dimensionality

    def supervised_model_predict(self, path_best_model, dataframe, model, target_name):
        if model != 'Ensemble':
            if model == 'Linear':
                model = \
                    self.load_pickle(os.path.join(path_best_model, "supervised_models",
                                                  target_name + '_' + model + "_model.pkl"))
            elif model == 'XGBoost':
                model = \
                    self.load_pickle(os.path.join(path_best_model, "supervised_models",
                                                  target_name + '_' + model + "_model.pkl"))
            return model.predict(dataframe)
        else:
            list_models_path = os.listdir(os.path.join(path_best_model, "supervised_models"))
            list_models = []
            for model in list_models_path:
                list_models.append(self.load_pickle(os.path.join(path_best_model, "supervised_models",
                                                                 target_name + '_' + model)))
            list_predicts = [model.predict(dataframe) for model in list_models]
            return [sum(sub_list) / len(sub_list) for sub_list in zip(*list_predicts)]

    @staticmethod
    def remove_files_from_folder(path_folder):
        for filename in os.listdir(path_folder):
            file_path = os.path.join(path_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def predict(self, dataframe, id_column, target_name, path_model):
        path_best_model = os.path.join('model_store', 'best_model')
        self.remove_files_from_folder(path_best_model)
        with zipfile.ZipFile(path_model, 'r') as zip_ref:
            zip_ref.extractall(path_best_model)

        best_parameters = self.load_pickle(os.path.join(path_best_model, target_name + "_best_parameters.pkl"))
        print(best_parameters)

        id_muestra = list(dataframe[id_column])
        dataframe_no_id = dataframe.drop(id_column, axis=1)
        dataframe_desired_columns = self.desired_columns_before_all(dataframe_no_id, path_best_model)
        dataframe_dummies = self.convert_dummies(dataframe_desired_columns, path_best_model)
        dataframe_no_duplicates = self.remove_duplicate_columns(dataframe_dummies)
        dataframe_no_constants = self.remove_constant_columns(dataframe_no_duplicates)
        dataframe_normalized = \
            self.normalized_columns(dataframe_no_constants, path_best_model)
        dataframe_no_nas = \
            self.imputer_predict(dataframe_normalized, path_best_model, best_parameters.get('algorithm_imput'))

        dataframe_feature_selection = \
            self.feature_selection_predict(dataframe_no_nas, path_best_model, best_parameters)

        if best_parameters.get('activated_pca'):
            dataframe_feature_selection = self.pca_predict(dataframe_feature_selection, path_best_model, best_parameters)
        res_predict = self.supervised_model_predict(
            path_best_model, dataframe_feature_selection, best_parameters.get('algorithm_supervised'), target_name)
        df_result = pd.DataFrame(columns=[id_column, 'Predict'])
        df_result[id_column] = id_muestra
        df_result['Predict'] = res_predict
        df_result.to_csv(os.path.join('Results', target_name + '_PredictedResults.csv'), index=False)
