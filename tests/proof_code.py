import pandas as pd
import os
import sys
import math
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.Ensemble.ensemble import Ensemble

filename = 'Dataset_Masterfile.xlsx'
sheet = '1_Var_CatYNum'

list_columns = [1]
sublist_1 = [i for i in range(61, 204)]
sublist_2 = [i for i in range(2550, 2616)]
list_columns.extend(sublist_1)
list_columns.extend(sublist_2)
print('Desired columns done')
if 'tests' in os.getcwd():
    xl_file = pd.ExcelFile(os.path.join('..', 'Data', filename))
else:
    xl_file = pd.ExcelFile(os.path.join('Data', filename))
dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
df = dfs[sheet]
target = list(df['DNAmGrimAge'])[:60]
print('Readed target')
algorithms_input = ['knn', 'linear', 'svm', 'xgboost']
threshold_variances = [0.01, 0.05, 0.07]
threshold_importances = [0.3, 0.5, 0.6]
algorithms_supervised = ['Linear', 'XGBoost', 'LightGBM', 'Ensemble']

seed = 42
ensemble_object = Ensemble(seed)

# Params predict ensemble: filename, sheet, list_columns, target, algorithm_input,
#                          threshold_variance=0.05, threshold_importance=0.3, seed=42,
#                          algorithm_supervised='Linear'
# Possible values:
#       algorithm_input: ['knn', 'linear', 'svm', 'xgboost', 'ensemble']
#       algorithm_supervised: ['Linear', 'XGBoost', 'LightGBM', 'Ensemble']

proof_one_model = True
if proof_one_model:
    rmse = ensemble_object.predict(filename=filename,
                                   sheet=sheet,
                                   list_columns=list_columns,
                                   target=target,
                                   algorithm_imput='knn',
                                   threshold_variance=0.05,
                                   threshold_importance=0.5,
                                   seed=42,
                                   algorithm_supervised='Linear'
                                   )
else:
    min_rmse = 1000
    for algorithm_supervised in algorithms_supervised:
        for algorithm_input in algorithms_input:
            for threshold_variance in threshold_variances:
                for threshold_importance in threshold_importances:
                    rmse = ensemble_object.predict(filename=filename,
                                                   sheet=sheet,
                                                   list_columns=list_columns,
                                                   target=target,
                                                   algorithm_imput=algorithm_input,
                                                   threshold_variance=threshold_variance,
                                                   threshold_importance=threshold_importance,
                                                   seed=42,
                                                   algorithm_supervised=algorithm_supervised
                                                   )
                    print('rmse')
                    print(rmse)
                    print('min_rmse')
                    print(min_rmse)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        best_parameters = {
                            'algorithm_input': algorithm_input,
                            'threshold_variance': threshold_variance,
                            'threshold_importance': threshold_importance,
                            'algorithm_supervised': algorithm_supervised
                        }
    # Launch the best
    print(best_parameters)
    rmse = ensemble_object.predict(filename=filename,
                                   sheet=sheet,
                                   list_columns=list_columns,
                                   target=target,
                                   algorithm_imput=best_parameters.get('algorithm_input'),
                                   threshold_variance=best_parameters.get('threshold_variance'),
                                   threshold_importance=best_parameters.get('threshold_importance'),
                                   seed=42,
                                   algorithm_supervised=best_parameters.get('algorithm_supervised')
                                   )
