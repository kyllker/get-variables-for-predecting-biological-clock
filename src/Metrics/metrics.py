import pandas as pd
import os
import sys
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)


class Metrics:
    def __init__(self, name_column_target):
        self.list_result = []
        self.name_column_target = name_column_target

    @staticmethod
    def round_2_decimals(number):
        return round(number, 2)

    def predict(self, list_result):
        self.list_result = list_result
        df_results = pd.DataFrame()
        df_results['ID'] = self.list_result[0]
        df_results['Predict'] = self.list_result[1]
        df_results['Predict'] = df_results['Predict'].apply(self.round_2_decimals)
        df_results['True'] = self.list_result[2]
        df_results['True'] = df_results['True'].apply(self.round_2_decimals)
        df_results = df_results.sort_values(by=['True'])
        print('Results shape')
        print(df_results.shape)
        df_results = df_results.round({'ID': 0, 'Predict': 2, 'True': 2})
        try:
            df_results.to_csv(os.path.join('Results', self.name_column_target + '_PredictedVsTrue.csv'), index=False)
        except FileNotFoundError:
            print('It is test mode')
        print(df_results)
        rmse = math.sqrt(mean_squared_error(list(df_results['Predict']), list(df_results['True'])))
        mae = mean_absolute_error(list(df_results['True']), list(df_results['Predict']))
        r2 = r2_score(list(df_results['True']), list(df_results['Predict']))
        errors = list()
        for i in range(len(list(df_results['Predict']))):
            err = (list(df_results['Predict'])[i] - list(df_results['True'])[i]) ** 2
            errors.append(err)
        f = pyplot.figure()
        pyplot.plot(errors)
        pyplot.xticks(ticks=[i for i in range(len(errors))], labels=list(df_results['Predict']))
        pyplot.xlabel('Predicted Value')
        pyplot.ylabel('Mean Squared Error')
        try:
            pyplot.savefig(os.path.join('Results', self.name_column_target + '_graphics.png'))
        except FileNotFoundError:
            print('It is test mode')
        f.clear()
        pyplot.close(f)

        return rmse, mae, r2, df_results






