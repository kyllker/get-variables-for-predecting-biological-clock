
import pandas as pd
from src.PCA.pca import PCAModel


class TestPCA:
    def setup_class(self):
        self.seed = 42
        self.pca_object = PCAModel(self.seed)
        self.df = pd.DataFrame([[1, 1, 1, 4], [2, 2, 2, 6], [3, 5, 3, 7]], columns=['ID', 'A', 'B', 'C'])

    def test_build_a_pca_object(self):
        assert isinstance(self.pca_object, PCAModel)

    def test_pca_predict(self):
        df_result, df_result_test = self.pca_object.predict(self.df, self.df, 'ID', 2)
        assert (df_result.shape[0] == 3) and (df_result.shape[1] == 3)
