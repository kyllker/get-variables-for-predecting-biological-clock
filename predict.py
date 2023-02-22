import pandas as pd
import os
import pickle
import warnings
from warnings import simplefilter
from src.Predictor.predictor import Predictor

warnings.filterwarnings('ignore')
simplefilter("ignore", category=RuntimeWarning)


class Predict:
    def __init__(self, df_predict, id_column, path_model):
        self.name_id_column = id_column
        self.df_predict = df_predict
        self.path_model = path_model

    def predict(self):
        predictor_object = Predictor()
        df_result = predictor_object.predict(self.df_predict, self.path_model)
        return df_result

