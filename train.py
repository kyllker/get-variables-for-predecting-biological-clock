import pandas as pd
import os
import sys
import pickle
import warnings
from warnings import simplefilter
from src.Ensemble.ensemble import Ensemble
import shutil

warnings.filterwarnings('ignore')
simplefilter("ignore", category=RuntimeWarning)


class Train:
    def __init__(self, list_columns, df_data, label_name, id_column, parameters, individual_all='Individual',
                 ids_test=[]):
        self.name_column_target = label_name
        self.name_id_column = id_column
        self.list_columns = list_columns
        self.df_data = df_data
        self.target = list(self.df_data[label_name])
        self.parameters = parameters
        self.individual_all = individual_all
        self.ids_test = ids_test

    def predict(self):
        seed = 42
        ensemble_object = Ensemble(seed, self.name_column_target)

        if self.individual_all == 'Individual':

            # Params predict ensemble: filename, sheet, list_columns, target, ids_test, algorithm_imput,
            #                          threshold_variance=0.05, threshold_importance=0.3, seed=42,
            #                          algorithm_supervised='Linear'
            # Possible values:
            #       algorithm_imput: ['mean_mode', 'knn', 'linear', 'svm', 'xgboost', 'ensemble']
            #       algorithm_supervised: ['Linear', 'XGBoost', 'LightGBM', 'Ensemble']

            rmse, mae, r2, df, best_5_features, train_rmse = ensemble_object.predict(
                df=self.df_data,
                list_columns=self.list_columns,
                target=self.target,
                id_column=self.name_id_column,
                ids_test=self.ids_test,
                algorithm_imput=self.parameters.get('algorithm_imput'),
                threshold_variance=self.parameters.get('threshold_variance'),
                threshold_importance=self.parameters.get('threshold_importance'),
                seed=42,
                algorithm_supervised=self.parameters.get('algorithm_supervised'),
                activated_pca=self.parameters.get('activated_pca'),
                n_components_pca=self.parameters.get('n_components_pca')
            )
            best_parameters = {
                'algorithm_imput': self.parameters.get('algorithm_imput'),
                'threshold_variance': self.parameters.get('threshold_variance'),
                'threshold_importance': self.parameters.get('threshold_importance'),
                'algorithm_supervised': self.parameters.get('algorithm_supervised'),
                'activated_pca': self.parameters.get('activated_pca'),
                'n_components_pca': self.parameters.get('n_components_pca')
            }
            with open(os.path.join('model_store', 'saved_models', self.name_column_target + '_best_parameters.pkl'),
                      'wb') as f:
                pickle.dump(best_parameters, f)

            shutil.make_archive(
                    os.path.join('model_store', 'compressed_model', self.name_column_target + '_best_model_' +
                                 str(round(rmse, 3)).replace('.', '_')), 'zip',
                    os.path.join('model_store', 'saved_models'))
            df_aux = df.sort_values('ID')
            print(df_aux)
            print(list(df_aux['Predict']))
            return rmse, mae, r2, df, best_5_features

        else:
            df_results = pd.DataFrame(0, index=[i for i in range(385)],
                                      columns=['Variance', 'Importance', 'Imputer', 'PCA', 'Supervised',
                                               'TrainRMSE', 'TestRMSE'])
            best_parameters = {}
            # algorithms_imput = ['mean_mode', 'knn', 'linear', 'svm', 'xgboost', 'ensemble']
            algorithms_imput = ['knn', 'linear', 'svm', 'xgboost']
            threshold_variances = [0.01, 0.05, 0.07]
            threshold_importances = [20, 30, 50, 1]
            algorithms_supervised = ['Linear', 'XGBoost', 'LightGBM', 'Ensemble']
            bool_pca = [False, True]
            # bool_pca = [True]
            ncomponents_pca = [5, 10, 20]
            min_rmse = 100
            cont = 0
            for act_pca in bool_pca:
                for algorithm_supervised in algorithms_supervised:
                    for algorithm_imput in algorithms_imput:
                        for threshold_variance in threshold_variances:
                            for threshold_importance in threshold_importances:
                                if act_pca:
                                    for ncom in ncomponents_pca:
                                        print(algorithm_supervised + ' - ' + algorithm_imput + ' - ' +
                                              str(threshold_variance) + ' - ' + str(threshold_importance) + ' - ' +
                                              str(ncom) + ' - ' + str(act_pca))
                                        rmse, mae, r2, df, best_5_features, train_rmse = ensemble_object.predict(
                                            df=self.df_data,
                                            list_columns=self.list_columns,
                                            target=self.target,
                                            id_column=self.name_id_column,
                                            ids_test=self.ids_test,
                                            algorithm_imput=algorithm_imput,
                                            threshold_variance=threshold_variance,
                                            threshold_importance=threshold_importance,
                                            seed=42,
                                            algorithm_supervised=algorithm_supervised,
                                            activated_pca=act_pca,
                                            n_components_pca=ncom
                                            )
                                        df_results.loc[cont, 'Variance'] = threshold_variance
                                        df_results.loc[cont, 'Importance'] = threshold_importance
                                        df_results.loc[cont, 'Imputer'] = algorithm_imput
                                        df_results.loc[cont, 'PCA'] = ncom
                                        df_results.loc[cont, 'Supervised'] = algorithm_supervised
                                        df_results.loc[cont, 'TrainRMSE'] = train_rmse
                                        df_results.loc[cont, 'TestRMSE'] = rmse
                                        cont = cont + 1
                                        if cont % 25 == 0:
                                            df_results.to_csv(
                                                os.path.join('model_store',
                                                             'results_' + self.name_column_target + '.csv'),
                                                index=False)
                                        print('rmse')
                                        print(rmse)
                                        print('min_rmse')
                                        print(min_rmse)
                                        if rmse < min_rmse:
                                            min_rmse = rmse
                                            best_parameters = {
                                                'algorithm_imput': algorithm_imput,
                                                'threshold_variance': threshold_variance,
                                                'threshold_importance': threshold_importance,
                                                'algorithm_supervised': algorithm_supervised,
                                                'activated_pca': act_pca,
                                                'n_components_pca': ncom
                                            }
                                            with open(os.path.join('model_store', 'saved_models',
                                                                   self.name_column_target + '_best_parameters.pkl'),
                                                      'wb') as f:
                                                pickle.dump(best_parameters, f)
                                            shutil.make_archive(
                                                os.path.join('model_store', 'compressed_model', self.name_column_target
                                                             + '_best_model_' + str(round(rmse, 3)).replace('.', '_')),
                                                'zip', os.path.join('model_store', 'saved_models'))
                                        print(best_parameters)
                                else:
                                    print(algorithm_supervised + ' - ' + algorithm_imput + ' - ' +
                                          str(threshold_variance) + ' - ' + str(threshold_importance) + ' - ' +
                                          str(0) + ' - ' + str(act_pca))
                                    rmse, mae, r2, df, best_5_features, train_rmse = ensemble_object.predict(
                                        df=self.df_data,
                                        list_columns=self.list_columns,
                                        target=self.target,
                                        id_column=self.name_id_column,
                                        ids_test=self.ids_test,
                                        algorithm_imput=algorithm_imput,
                                        threshold_variance=threshold_variance,
                                        threshold_importance=threshold_importance,
                                        seed=42,
                                        algorithm_supervised=algorithm_supervised,
                                        activated_pca=act_pca,
                                        n_components_pca=0
                                    )
                                    df_results.loc[cont, 'Variance'] = threshold_variance
                                    df_results.loc[cont, 'Importance'] = threshold_importance
                                    df_results.loc[cont, 'Imputer'] = algorithm_imput
                                    df_results.loc[cont, 'PCA'] = 'False'
                                    df_results.loc[cont, 'Supervised'] = algorithm_supervised
                                    df_results.loc[cont, 'TrainRMSE'] = train_rmse
                                    df_results.loc[cont, 'TestRMSE'] = rmse
                                    if cont % 25 == 0:
                                        df_results.to_csv(
                                            os.path.join('model_store',
                                                         'results_' + self.name_column_target + '.csv'), index=False)
                                    cont = cont + 1
                                    print('rmse')
                                    print(rmse)
                                    print('min_rmse')
                                    print(min_rmse)
                                    if rmse < min_rmse:
                                        min_rmse = rmse
                                        best_parameters = {
                                            'algorithm_imput': algorithm_imput,
                                            'threshold_variance': threshold_variance,
                                            'threshold_importance': threshold_importance,
                                            'algorithm_supervised': algorithm_supervised,
                                            'activated_pca': act_pca,
                                            'n_components_pca': 0
                                        }
                                        with open(os.path.join('model_store', 'saved_models',
                                                               self.name_column_target + '_best_parameters.pkl'),
                                                  'wb') as f:
                                            pickle.dump(best_parameters, f)
                                        shutil.make_archive(
                                            os.path.join('model_store', 'compressed_model', self.name_column_target +
                                                         '_best_model_' +
                                                         str(round(rmse, 3)).replace('.', '_')),
                                            'zip', os.path.join('model_store', 'saved_models'))
                                    print(best_parameters)
            df_results.to_csv(
                os.path.join('model_store',
                             'results' + self.name_column_target + '.csv'), index=False)
            print(df['Predict'])
            return rmse, mae, r2, df, best_5_features
