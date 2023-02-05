import pandas as pd
import os
import pickle
import warnings
from warnings import simplefilter
from src.Predictor.predictor import Predictor

warnings.filterwarnings('ignore')
simplefilter("ignore", category=RuntimeWarning)

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

xl_file = pd.ExcelFile(os.path.join('Data', filename))

dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}

dfs[sheet].iloc[:60, :]


print('Readed target')

ids_test = [15, 23, 34, 52, 48, 44, 42, 21, 45, 60, 6, 5]

df_to_predict = df.filter(items=ids_test, axis=0)

predictor_object = Predictor()

path_model = os.path.join('src', 'model_store', 'compressed_model', 'best_model.zip')
predictor_object.predict(df_to_predict, path_model)

