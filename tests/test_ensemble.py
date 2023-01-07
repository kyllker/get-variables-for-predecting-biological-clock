import os
import sys
import numpy as np
import pandas as pd
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.Ensemble.ensemble import Ensemble


class TestEnsemble:
    def setup_class(self):
        self.seed = 42
        self.ensemble_object = Ensemble(self.seed)

        self.df = pd.DataFrame([[1, 1], [1, 2]], columns=['A', 'B'])

        self.df_no_numerical = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['A', 'B'])

    def test_build_an_ensemble_object(self):
        assert isinstance(self.ensemble_object, Ensemble)