import pandas as pd
import os
import sys
import math
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)


class Metrics:
    def __init__(self):
        self.list_result = []

    @staticmethod
    def round_2_decimals(number):
        return round(number, 2)

    def predict(self, list_result, algorithm_supervised='Linear'):
        self.list_result = list_result
        df_results = pd.DataFrame()
        df_results['Predict'] = self.list_result[0]
        df_results['Predict'] = df_results['Predict'].apply(self.round_2_decimals)
        df_results['True'] = self.list_result[1]
        df_results['True'] = df_results['True'].apply(self.round_2_decimals)
        df_results = df_results.sort_values(by=['Predict'])
        print('Results shape')
        print(df_results.shape)
        df_results.to_csv(os.path.join('src', 'Results', 'PredictedVsTrue.csv'), index=False)
        print(df_results)
        rmse = math.sqrt(mean_squared_error(list(df_results['Predict']), list(df_results['True'])))
        print('El rmse entre el ' + algorithm_supervised + ' model y el reloj biológico DNAmGrimAge es: ' + str(rmse))
        errors = list()
        for i in range(len(list(df_results['Predict']))):
            err = (list(df_results['Predict'])[i] - list(df_results['True'])[i]) ** 2
            errors.append(err)
        pyplot.plot(errors)
        pyplot.xticks(ticks=[i for i in range(len(errors))], labels=list(df_results['Predict']))
        pyplot.xlabel('Predicted Value')
        pyplot.ylabel('Mean Squared Error')
        pyplot.savefig(os.path.join('src', 'Results', 'graphics.png'))

        return rmse






