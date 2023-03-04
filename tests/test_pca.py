import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.PCA.pca import PCAModel


class TestPCA:
    def setup_class(self):
        self.seed = 42
        self.pca_object = PCAModel(self.seed)
        self.df = pd.DataFrame([[1, 1, 1, 4], [2, 2, 2, 6], [3, 5, 3, 7]], columns=['ID', 'A', 'B', 'C'])

    def test_build_a_pca_object(self):
        assert isinstance(self.pca_object, PCAModel)

    def test_pca_predict(self):
        df_result = self.pca_object.predict(self.df, 'ID', 2)
        assert (df_result.shape[0] == 3) and (df_result.shape[1] == 3)