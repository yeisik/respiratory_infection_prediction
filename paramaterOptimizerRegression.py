import time
import numpy as np
import sys
import optuna
from scipy.stats import pearsonr

from sklearn.linear_model import Ridge,ElasticNet,Lasso,BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import LeaveOneOut
import warnings


# REGRESSION OPTIMIZATION

warnings.simplefilter("ignore")

def ParamaterOpt(algorithm, model, trainInputs,trainTargets):
    def optimizeLOO(trial):

        # REGRESSORS : 
        if algorithm == "LinearSVR":
            params = {
                'epsilon': trial.suggest_float('epsilon', 0, 1),
                'C': trial.suggest_float('C', 1e-4, 1e3),
                'max_iter': trial.suggest_categorical('max_iter', [100000])
            }
    
        if algorithm == "BayesRidge":
            params = {
                'alpha_1': trial.suggest_float('alpha_1', 1e-7, 1),
                'alpha_2': trial.suggest_float('alpha_2', 1e-7, 1),
                'lambda_1': trial.suggest_float('lambda_1', 1e-7, 1),
                'lambda_2': trial.suggest_float('lambda_2', 1e-7, 1),
                'n_iter': trial.suggest_categorical('n_iter', [100000])
            }
        #Ridge
        if algorithm == "Ridge":

            params = {
                'alpha': trial.suggest_float('alpha', 0.000001, 10),
                'solver': trial.suggest_categorical('solver', ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                'max_iter': trial.suggest_categorical('max_iter', [10000])
            }

        if algorithm == "ElasticNet":
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 2),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.05, 1),
                'max_iter': trial.suggest_categorical('max_iter', [100000])
            }

        if algorithm == "Lasso":
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 2),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
                'max_iter': trial.suggest_categorical('max_iter', [200000])
            }
            
        if algorithm == "GradientR":
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'subsample': trial.suggest_float('subsample', 0.01, 1.0),
                'max_depth': trial.suggest_int('max_depth', 2, 100)
            }
            
        if algorithm == "RandomForestRegressor":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000)
            }
            
        if algorithm == "KNNR":
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 2, len(X)-1),
                'weights': trial.suggest_categorical('weights', ['uniform','distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute']),
                'leaf_size': trial.suggest_int('leaf_size', 10, 100),
                'p': trial.suggest_int('p', 1, 3)
            }

        if algorithm == "DTreeR":
            params = {
                'splitter': trial.suggest_categorical('splitter',
                                                       ['best', 'random'])
            }

        if algorithm == "XGBR":
            params = {
                'booster': trial.suggest_categorical('booster', ['gbtree','gblinear','dart']),
                'verbosity': trial.suggest_categorical('verbosity', [0]),
                'eta': trial.suggest_float('eta', 0.001, 1),
                'gamma': trial.suggest_float('gamma', 0, 100),
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'subsample': trial.suggest_float('subsample', 0.0001, 1),
                'lambda': trial.suggest_float('lambda', 0.01, 5),
                'alpha': trial.suggest_float('alpha', 0.01, 5)
            }

        _predicted = []
        _true = []

        for train_index, test_index in loot.split(X):

            trainInputs, valInputs = X[train_index], X[test_index]
            trainTargets, valTargets = y[train_index], y[test_index]

            try:
                model.set_params(**params)
                model.fit(trainInputs, trainTargets)
                _predicted.append(model.predict(valInputs)[0])
                _true.append(valTargets[0])

            except Exception as e:
                print("HATA ",str(e))
                continue
                
        pscore = pearsonr(_true, _predicted)
        return pscore[0]


    st = time.time()
    X = trainInputs
    y = trainTargets
    loot = LeaveOneOut()
    optuna.logging.disable_default_handler()
    study = optuna.create_study(direction='maximize')
    study.optimize(optimizeLOO, n_trials=5)

    #print("Optimization Working Time :",time.time()-st)
    
    return study.best_trial
