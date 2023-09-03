from sklearn.decomposition import PCA as sk_pca
import pandas as pd
import pickle
import os


class PCAModel:

    def __init__(self, seed):
        self.seed = seed

    def predict(self, x_train, x_test, id_column, ncomponents=20):
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        edad_cronologica = True
        if 'Edad_Cronologica' in x_train.columns.values.tolist():
            x_train_no_edad = x_train.drop(columns=[id_column, 'Edad_Cronologica'])
            x_test_no_edad = x_test.drop(columns=[id_column, 'Edad_Cronologica'])
        else:
            x_train_no_edad = x_train.drop(columns=[id_column])
            x_test_no_edad = x_test.drop(columns=[id_column])
            edad_cronologica = False
        if x_train_no_edad.shape[1] <= ncomponents:
            return x_train, x_test
        else:
            pca = sk_pca(n_components=ncomponents, random_state=self.seed)
            array_pca = pca.fit_transform(x_train_no_edad)
            x_test_pca = pd.DataFrame(pca.transform(x_test_no_edad))
            pca_columns = x_train_no_edad.columns.values.tolist()
            print(os.getcwd())
            try:
                with open(os.path.join('model_store', 'saved_models', 'pca', 'pca_model_' + str(ncomponents) + '.pkl'),
                          'wb') as f:
                    pickle.dump([pca, pca_columns], f)
            except:
                os.mkdir(os.path.join('model_store', 'saved_models', 'pca'))
                with open(os.path.join('model_store', 'saved_models', 'pca', 'pca_model_' + str(ncomponents) + '.pkl'),
                          'wb') as f:
                    pickle.dump([pca, pca_columns], f)
            df_pca = pd.DataFrame(array_pca)
            df_pca_test = pd.DataFrame(array_pca)
            if edad_cronologica:
                df_reduced_dimensionality = pd.concat(
                    [x_train.loc[:, [id_column, 'Edad_Cronologica']], df_pca], 1)
                df_reduced_dimensionality_test = pd.concat(
                    [x_test.loc[:, [id_column, 'Edad_Cronologica']], x_test_pca], 1)
            else:
                df_reduced_dimensionality = pd.concat(
                    [x_train.loc[:, [id_column]], df_pca], 1)
                df_reduced_dimensionality_test = pd.concat(
                    [x_test.loc[:, [id_column]], df_pca_test], 1)
            return df_reduced_dimensionality, df_reduced_dimensionality_test
