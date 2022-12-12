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
    xl_file = pd.ExcelFile(os.path.join('..', 'src', 'Data', filename))
else:
    xl_file = pd.ExcelFile(os.path.join('src', 'Data', filename))
dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
df = dfs[sheet]
target = list(df['DNAmGrimAge'])[:60]
print('readed target')

algorithms_imput = ['knn', 'linear', 'svm', 'xgboost']
threshold_variances = [0.01, 0.05, 0.07]
threshold_importances = [0.3, 0.5, 0.6]
algorithms_supervised = ['Linear', 'XGBoost', 'Ensemble']

ensemble_object = Ensemble()

# Params predict ensemble: filename, sheet, list_columns, target, algorithm_imput,
#                          threshold_variance=0.05, threshold_importance=0.3, seed=42,
#                          algorithm_supervised='Linear'
# Possible values:
#       algorithm_imput: ['knn', 'linear', 'svm', 'xgboost']
#       algorithm_supervised: ['Linear', 'XGBoost', 'Ensemble']

proof_one_model = False
if proof_one_model:
    list_res = ensemble_object.predict(filename=filename,
                                     sheet=sheet,
                                     list_columns=list_columns,
                                     target=target,
                                     algorithm_imput='knn',
                                     threshold_variance=0.05,
                                     threshold_importance=0.5,
                                     seed=42,
                                     algorithm_supervised='Linear'
                                     )
    results = pd.DataFrame()
    results['Predict'] = list_res[0]
    results['True'] = list_res[1]
    print(results)
    rmse = math.sqrt(mean_squared_error(list(results['Predict']), list(results['True'])))
    print('El rmse entre el linear model y el reloj biológico DNAmGrimAge es: ' + str(rmse))

else:
    min_rmse = 1000
    for algorithm_supervised in algorithms_supervised:
        for algorithm_imput in algorithms_imput:
            for threshold_variance in threshold_variances:
                for threshold_importance in threshold_importances:
                    list_res = ensemble_object.predict(filename=filename,
                                                       sheet=sheet,
                                                       list_columns=list_columns,
                                                       target=target,
                                                       algorithm_imput=algorithm_imput,
                                                       threshold_variance=threshold_variance,
                                                       threshold_importance=threshold_importance,
                                                       seed=42,
                                                       algorithm_supervised=algorithm_supervised
                                                       )
                    results = pd.DataFrame()
                    results['Predict'] = list_res[0]
                    results['True'] = list_res[1]
                    rmse = math.sqrt(mean_squared_error(list(results['Predict']), list(results['True'])))
                    print('rmse')
                    print(rmse)
                    print('min_rmse')
                    print(min_rmse)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        best_parameters = {
                            'algorithm_imput': algorithm_imput,
                            'threshold_variance': threshold_variance,
                            'threshold_importance': threshold_importance,
                            'algorithm_supervised': algorithm_supervised
                        }
    # Launch the best
    list_res = ensemble_object.predict(filename=filename,
                                       sheet=sheet,
                                       list_columns=list_columns,
                                       target=target,
                                       algorithm_imput=best_parameters.get('algorithm_imput'),
                                       threshold_variance=best_parameters.get('threshold_variance'),
                                       threshold_importance=best_parameters.get('threshold_importance'),
                                       seed=42,
                                       algorithm_supervised=best_parameters.get('algorithm_supervised')
                                       )
    results = pd.DataFrame()
    results['Predict'] = list_res[0]
    results['True'] = list_res[1]
    print(results)
    rmse = math.sqrt(mean_squared_error(list(results['Predict']), list(results['True'])))
    print('El rmse entre el linear model y el reloj biológico DNAmGrimAge es: ' + str(rmse))
