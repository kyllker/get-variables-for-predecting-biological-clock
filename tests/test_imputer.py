import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)

from src.Preprocessors.imputer import Imputer


class TestImputer:
    def setup_class(self):
        self.seed = 42
        self.imputer_object = Imputer(self.seed)

        self.df = pd.DataFrame([[1, 1], [1, 2]], columns=['A', 'B'])

        self.df_na = pd.DataFrame([[1, np.nan], [1, 2]], columns=['A', 'B'])

        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])

        self.iris = load_iris()

    def test_build_an_imputer_object(self):
        assert isinstance(self.imputer_object, Imputer)

    def test_check_there_are_na_values_false(self):
        assert not self.imputer_object.check_there_are_na_values(self.df)

    def test_check_there_are_na_values_true(self):
        assert self.imputer_object.check_there_are_na_values(self.df_na)

    def test_get_na_and_no_na_columns(self):
        assert self.imputer_object.get_na_and_no_na_columns(self.df_na)[0][0] == 'B' and \
               self.imputer_object.get_na_and_no_na_columns(self.df_na)[1][0] == 'A'
