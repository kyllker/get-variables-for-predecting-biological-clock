import pandas as pd
import os
import sys
import math
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)
project_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_dir)
from src.Ensemble.ensemble import Ensemble

filename = 'Dataset_Masterfile.xlsx'
sheet = '1_Var_CatYNum'

list_columns = [0, 1]
sublist_1 = [i for i in range(61, 204)]
sublist_2 = [i for i in range(2550, 2616)]
list_columns.extend(sublist_1)
list_columns.extend(sublist_2)
print('Desired columns done')


xl_file = pd.ExcelFile(os.path.join('Data', filename))

dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
df = dfs[sheet]
target = list(df['DNAmGrimAge'])[:60]

print('Readed target')

algorithms_imput = ['mean_mode', 'knn', 'linear', 'svm', 'xgboost', 'ensemble']
threshold_variances = [0.01, 0.05, 0.07]
threshold_importances = [20, 30, 50]
algorithms_supervised = ['Linear', 'XGBoost', #'LightGBM',
                         'Ensemble']
bool_pca = [True, False]
ncomponents_pca = [5, 10, 20, 50]

seed = 42
ensemble_object = Ensemble(seed)

# Params predict ensemble: filename, sheet, list_columns, target, ids_test, algorithm_imput,
#                          threshold_variance=0.05, threshold_importance=0.3, seed=42,
#                          algorithm_supervised='Linear'
# Possible values:
#       algorithm_imput: ['mean_mode', 'knn', 'linear', 'svm', 'xgboost', 'ensemble']
#       algorithm_supervised: ['Linear', 'XGBoost', 'LightGBM', 'Ensemble']

proof_one_model = True
if proof_one_model:
    rmse = ensemble_object.predict(filename=filename,
                                   sheet=sheet,
                                   list_columns=list_columns,
                                   target=target,
                                   ids_test=[15, 23, 34, 52, 48, 44, 42, 21, 45, 60, 6, 5],
                                   algorithm_imput='xgboost',
                                   threshold_variance=0.01,
                                   threshold_importance=50,
                                   seed=42,
                                   algorithm_supervised='Linear',
                                   activated_pca=True,
                                   n_components_pca=10
                                   )
else:
    min_rmse = 1000
    for act_pca in [True, False]:
        for ncom in ncomponents_pca:
            for algorithm_supervised in algorithms_supervised:
                for algorithm_imput in algorithms_imput:
                    for threshold_variance in threshold_variances:
                        for threshold_importance in threshold_importances:
                            print(algorithm_supervised + ' - ' + algorithm_imput + ' - ' + str(threshold_variance) +
                                  ' - ' + str(threshold_importance) + ' - ' + str(ncom) + ' - ' + str(act_pca))
                            rmse = ensemble_object.predict(filename=filename,
                                                           sheet=sheet,
                                                           list_columns=list_columns,
                                                           target=target,
                                                           ids_test=[15, 23, 34, 52, 48, 44, 42, 21, 45, 60, 6, 5],
                                                           algorithm_imput=algorithm_imput,
                                                           threshold_variance=threshold_variance,
                                                           threshold_importance=threshold_importance,
                                                           seed=42,
                                                           algorithm_supervised=algorithm_supervised,
                                                           activated_pca=act_pca,
                                                           n_components_pca=ncom
                                                           )
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
                                    'algorithm_supervised': algorithm_supervised,
                                    'activated_pca': act_pca,
                                    'n_components_pca': ncom
                                }
                            print(best_parameters)
            # Launch the best
    print(best_parameters)
    rmse = ensemble_object.predict(filename=filename,
                                   sheet=sheet,
                                   list_columns=list_columns,
                                   target=target,
                                   ids_test=[15, 23, 34, 52, 48, 44, 42, 21, 45, 60, 6, 5],
                                   algorithm_imput=best_parameters.get('algorithm_imput'),
                                   threshold_variance=best_parameters.get('threshold_variance'),
                                   threshold_importance=best_parameters.get('threshold_importance'),
                                   seed=42,
                                   algorithm_supervised=best_parameters.get('algorithm_supervised'),
                                   activated_pca=best_parameters.get('activated_pca'),
                                   n_components_pca=best_parameters.get('n_components_pca')
                                   )
