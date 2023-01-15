import time
import numpy as np
import sys
import optuna
import math
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier as XGB

from sklearn.model_selection import LeaveOneOut
import warnings

warnings.simplefilter("ignore")



def ParamaterOpt(algorithm, model, trainInputs,trainTargets):
    
    def optimizeLOO(trial):
        
        if algorithm == "LR":
                     
            params = {
                'C':trial.suggest_float('C', math.pow(2,-12), math.pow(2,12)),
                'solver':trial.suggest_categorical('solver', ['liblinear','lbfgs','newton-cg','saga'])
            }

        if algorithm == "SVM":
            
            params = {
                'C':trial.suggest_float('C', math.pow(2,-12), math.pow(2,12)),
                'kernel':trial.suggest_categorical('kernel', ['linear','poly','rbf']),
                'gamma':trial.suggest_float('gamma', 0.000001,10),
				'probability':trial.suggest_categorical('probability', [True])}
        
        if algorithm == "RF":
            
            params = {
                'n_estimators':trial.suggest_int('n_estimators',2, 1000),
                'criterion':trial.suggest_categorical('criterion', ['gini','entropy']),
                'min_samples_leaf':trial.suggest_int('min_samples_leaf',1, 32)}

        if algorithm == "KNN":
            
            params = {
                'n_neighbors':trial.suggest_int('n_neighbors',2, len(X)-1),
                'algorithm':trial.suggest_categorical('algorithm', ['auto','ball_tree','kd_tree','brute']),
                'leaf_size':trial.suggest_int('leaf_size',2,1000)}

        if algorithm == "XGB":
            
            params = {
                'verbosity':trial.suggest_categorical('verbosity', [0]),
                'objective':trial.suggest_categorical('objective', ['binary:logistic']),
                'eval_metric':trial.suggest_categorical('eval_metric', ['error']),
                'booster':trial.suggest_categorical('loss', ['gbtree', 'gblinear']),

                'n_estimators':trial.suggest_int('n_estimators',2, 1000),
                'max_depth':trial.suggest_int('max_depth', 2, 512),

                'learning_rate':trial.suggest_float('learning_rate', 0.0001, 10),
                'gamma':trial.suggest_float('gamma', 0, 100),
                'min_child_weight':trial.suggest_float('min_child_weight', 1, 32),
                'subsample':trial.suggest_float('subsample', 0, 1),
                'colsample_bytree':trial.suggest_float('colsample_bytree', 0, 1),
                'reg_alpha':trial.suggest_float('reg_alpha', 0, 32),
                'reg_lambda':trial.suggest_float('reg_lambda', 0, 32) }


        _totalscore = 0

        for train_index, test_index in loot.split(X):


            trainInputs, valInputs = X[train_index], X[test_index]
            trainTargets, valTargets = y[train_index], y[test_index]

            try:
                model.set_params(**params)
                model.fit(trainInputs, trainTargets)

                prediction= model.score(valInputs,valTargets)
                if prediction == valTargets[0]:                   
                    _totalscore = _totalscore + 1.

            except Exception as e:
                continue

        accscore = _totalscore / float(len(X))       
        #accscore = accuracy_score(_true, _predicted)
        return accscore
        
    X,y = trainInputs,np.array(trainTargets)

    loot = LeaveOneOut()
    optuna.logging.disable_default_handler()
    study = optuna.create_study(direction='maximize')
    study.optimize(optimizeLOO, n_trials=5)
    return study.best_trial
