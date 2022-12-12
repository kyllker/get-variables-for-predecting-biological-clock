import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb


class SupervisedModel:

    def __init__(self):
        pass

    @staticmethod
    def linear_model(X_train, y_train, seed=42):
        folds = KFold(n_splits = 10, shuffle = True, random_state = seed)
        hyper_params = [{'n_features_to_select': list(range(1, 20))}]
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        rfe = RFE(lm)
        model_cv = GridSearchCV(estimator = rfe,
                        param_grid = hyper_params,
                        scoring= 'r2',
                        cv = folds,
                        verbose = 1,
                        return_train_score=True)
        model_cv.fit(X_train, y_train)
        return model_cv

    @staticmethod
    def xgboost_model(X_train, y_train, seed=42):
        param_dist = {'n_estimators': stats.randint(80, 150),
                      'learning_rate': [0.001, 0.01, 0.1],
                      'subsample': stats.uniform(0.3, 0.7),
                      'max_depth': [3, 6, 9],
                      'colsample_bytree': stats.uniform(0.5, 0.45),
                      'min_child_weight': [1, 3]
                      }
        clf_xgb = xgb.XGBRegressor(objective='reg:squarederror')
        clf = RandomizedSearchCV(clf_xgb, param_distributions=param_dist, n_iter=25, scoring='neg_mean_squared_error',
                                 error_score=0, verbose=3, n_jobs=-1)
        numFolds = 10
        folds = KFold(n_splits=numFolds, shuffle=True)
        best_estimator = ['']
        current_mse = 2000
        for train_index, test_index in folds.split(X_train):
            X_train_model, X_test = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_train_model, y_test = y_train[train_index], y_train[test_index]
            clf.fit(X_train_model, list(y_train_model))
            if mean_squared_error(y_test, clf.predict(X_test)) < current_mse:
                best_estimator[0] = clf.best_estimator_
        clf = best_estimator[0]
        clf.fit(X_train, list(y_train))
        return clf

    def predict(self, X_train, y_train, X_test, seed, algorithm='Linear'):
        if algorithm == 'Linear':
            model = self.linear_model(X_train, y_train, seed)
            return model.predict(X_test)
        elif algorithm == 'XGBoost':
            model = self.xgboost_model(X_train, y_train, seed)
            return model.predict(X_test)
        elif algorithm == 'Ensemble':
            model_linear = self.linear_model(X_train, y_train, seed)
            model_xgboost = self.xgboost_model(X_train, y_train, seed)
            res_linear = model_linear.predict(X_test)
            res_xgboost = model_xgboost.predict(X_test)
            return [(g + h) / 2 for g, h in zip(res_linear, res_xgboost)]
