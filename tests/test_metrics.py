import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Load src folder (in all cases)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)

from src.Metrics.metrics import Metrics


class TestMetrics:
    def setup_class(self):
        self.seed = 42
        self.metric_object = Metrics('ID')

    def test_build_a_metric_object(self):
        assert isinstance(self.metric_object, Metrics)

    def test_round_2_decimals(self):
        assert self.metric_object.round_2_decimals(3.568) == 3.57

    def test_predict(self):
        list_result = [[1, 2, 3], [3, 4, 6], [3.5, 4.2, 6]]
        rmse, mae, r2, df_results = self.metric_object.predict(list_result)
        assert (rmse == 0.3109126351029605) and (mae == 0.2333333333333334) and (r2 == 0.9128256513026052)

