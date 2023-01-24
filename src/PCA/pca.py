from sklearn.decomposition import PCA as sk_pca
import pandas as pd


class PCAModel:

    def __init__(self, seed):
        self.seed = seed

    def predict(self, x_train, ncomponents=20):
        x_train_no_edad = x_train.drop(columns=['ID_Muestra', 'Edad_Cronologica'])
        if x_train.shape[1] <= 50:
            return x_train
        else:
            pca = sk_pca(n_components=ncomponents, random_state=self.seed)
            array_pca = pca.fit_transform(x_train_no_edad)
            df_pca = pd.DataFrame(array_pca)
            df_reduced_dimensionality = pd.concat(
                [x_train.loc[:, ['ID_Muestra', 'Edad_Cronologica']], df_pca], 1)
            return df_reduced_dimensionality
