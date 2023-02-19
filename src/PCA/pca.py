from sklearn.decomposition import PCA as sk_pca
import pandas as pd
import pickle
import os


class PCAModel:

    def __init__(self, seed):
        self.seed = seed

    def predict(self, x_train, id_column, ncomponents=20):
        edad_cronologica = True
        if 'Edad_Cronologica' in x_train.columns.values.tolist():
            x_train_no_edad = x_train.drop(columns=[id_column, 'Edad_Cronologica'])
        else:
            x_train_no_edad = x_train.drop(columns=[id_column])
            edad_cronologica = False
        if x_train_no_edad.shape[1] <= ncomponents:
            return x_train
        else:
            pca = sk_pca(n_components=ncomponents, random_state=self.seed)
            array_pca = pca.fit_transform(x_train_no_edad)
            pca_columns = x_train_no_edad.columns.values.tolist()
            with open(os.path.join('src', 'model_store', 'saved_models', 'pca', 'pca_model.pkl'), 'wb') as f:
                pickle.dump([pca, pca_columns], f)
            df_pca = pd.DataFrame(array_pca)
            if edad_cronologica:
                df_reduced_dimensionality = pd.concat(
                    [x_train.loc[:, [id_column, 'Edad_Cronologica']], df_pca], 1)
            else:
                df_reduced_dimensionality = pd.concat(
                    [x_train.loc[:, [id_column]], df_pca], 1)
            return df_reduced_dimensionality
