
import os
import warnings
from warnings import simplefilter
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import pickle
simplefilter("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore')


class SupervisedModel:

    def __init__(self, seed, name_column_target):
        self.seed = seed
        self.name_column_target = name_column_target

    def linear_model(self, x_train, y_train):
        folds = KFold(n_splits=10, shuffle=True, random_state=self.seed)
        hyper_params = [{'n_features_to_select': list(range(1, 20))}]
        lm = LinearRegression()
        lm.fit(x_train, y_train)
        rfe = RFE(lm)
        model_cv = GridSearchCV(estimator=rfe,
                                param_grid=hyper_params,
                                scoring='r2',
                                cv=folds,
                                verbose=0,
                                return_train_score=True
                                )
        model_cv.fit(x_train, y_train)
        with open(os.path.join('model_store', 'saved_models', 'supervised_models',
                               self.name_column_target + '_Linear_model.pkl'), 'wb') as f:
            pickle.dump(model_cv, f)
        return model_cv

    def xgboost_model(self, x_train, y_train):
        x_train = x_train.reset_index(drop=True)
        x_train = x_train.astype(float)
        y_train = y_train.reset_index(drop=True)
        param_dist = {'n_estimators': stats.randint(80, 150),
                      'learning_rate': [0.001, 0.01, 0.1],
                      'subsample': stats.uniform(0.3, 0.7),
                      'max_depth': [3, 6, 9],
                      'colsample_bytree': stats.uniform(0.5, 0.45),
                      'min_child_weight': [1, 3],
                      'seed': [self.seed]
                      }
        clf_xgb = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0, random_state=self.seed)
        clf = RandomizedSearchCV(clf_xgb, param_distributions=param_dist, n_iter=25, scoring='neg_mean_squared_error',
                                 error_score=0, verbose=0, n_jobs=-1, random_state=self.seed)
        num_folds = 10
        folds = KFold(n_splits=num_folds, shuffle=True)
        best_estimator = ['']
        current_mse = 2000
        for train_index, test_index in folds.split(x_train):
            x_train_model, x_test = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
            y_train_model = [y_train[i] for i in train_index]
            y_test = [y_train[i] for i in test_index]
            clf.fit(x_train_model, list(y_train_model))
            if mean_squared_error(y_test, clf.predict(x_test)) < current_mse:
                best_estimator[0] = clf.best_estimator_
        clf = best_estimator[0]
        clf.fit(x_train, list(y_train))
        with open(os.path.join('model_store', 'saved_models', 'supervised_models',
                               self.name_column_target + '_XGBoost_model.pkl'), 'wb') as f:
            pickle.dump(clf, f)
        return clf

    def lightgbm_model(self, x_train, y_train):
        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        x_train = x_train.astype(float)
        param_dist = {
            'task': ['train'],
            'boosting_type': ['gbdt'],
            'objective': ['regression'],
            'num_leaves': [2, 3, 5],
            'learning_rate': [0.001, 0.01, 0.1],
            'metric': ['l2', 'l1'],
            'seed': [self.seed],
            'verbosity': [-1]
        }
        clf_lgb = lgb.LGBMRegressor(verbosity=-1, silent=True, random_state=self.seed)
        clf = RandomizedSearchCV(estimator=clf_lgb,
                                 param_distributions=param_dist, cv=5,
                                 n_iter=100,
                                 verbose=-1,
                                 random_state=self.seed
                                 )

        num_folds = 10
        folds = KFold(n_splits=num_folds, shuffle=True)
        best_estimator = ['']
        current_mse = 2000
        for train_index, test_index in folds.split(x_train):
            x_train_model, x_test = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
            y_train_model = [y_train[i] for i in train_index]
            y_test = [y_train[i] for i in test_index]
            clf.fit(x_train_model, list(y_train_model))
            if mean_squared_error(y_test, clf.predict(x_test)) < current_mse:
                best_estimator[0] = clf.best_estimator_
        clf = best_estimator[0]
        clf.fit(x_train, list(y_train))
        with open(os.path.join('model_store', 'saved_models', 'supervised_models',
                               self.name_column_target + '_LightGBM_model.pkl'), 'wb') as f:
            pickle.dump(clf, f)
        return clf

    def predict(self, x_train, y_train, x_test, id_column, seed=42, algorithm='Linear'):
        x_train_no_id = x_train.drop(id_column, axis=1)
        id_muestra_test = list(x_test[id_column])
        x_test_no_id = x_test.drop(id_column, axis=1)
        x_train_no_id = x_train_no_id.astype(float)
        x_test_no_id = x_test_no_id.astype(float)

        if algorithm == 'Linear':
            model = self.linear_model(x_train_no_id, y_train)
            return [id_muestra_test, list(model.predict(x_test_no_id))]
        elif algorithm == 'XGBoost':
            model = self.xgboost_model(x_train_no_id, y_train)
            return [id_muestra_test, list(model.predict(x_test_no_id))]
        elif algorithm == 'LightGBM':
            model = self.lightgbm_model(x_train_no_id, y_train)
            return [id_muestra_test, list(model.predict(x_test_no_id))]
        elif algorithm == 'Ensemble':
            model_linear = self.linear_model(x_train_no_id, y_train)
            model_xgboost = self.xgboost_model(x_train_no_id, y_train)
            # model_lightgbm = self.lightgbm_model(x_train_no_id, y_train)
            res_linear = model_linear.predict(x_test_no_id)
            res_xgboost = model_xgboost.predict(x_test_no_id)
            # res_lightgbm = model_lightgbm.predict(x_test_no_id)
            # predictions = [(g + h + j) / 3 for g, h, j in zip(res_linear, res_xgboost, res_lightgbm)]
            prediction_linear_xg = [(g + h) / 2 for g, h in zip(res_linear, res_xgboost)]
            return [id_muestra_test, prediction_linear_xg]
