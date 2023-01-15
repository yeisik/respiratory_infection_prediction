from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

# NUMPY VERSION '1.22.4'
import numpy as np
import pandas as pd
import json
import ast


# SKLEARN VERSION : '1.1.1'
# CLASSIFIERS
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier as XGB

# REGRESSORS
from sklearn.linear_model import ElasticNet,Lasso,Ridge,BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#METRICS
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

#Parameter Optimization Classes
from paramaterOptimizerRegression import ParamaterOpt as OP_regression
from paramaterOptimizerClassification import ParamaterOpt as OP_classification

#ignore all warnings
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1064)

_datasetPath = "datasets"
experimentList = ['ALL','DEE1_RSV',  'DEE2_H3N2',  'DEE3_H1N1',  'DEE4X_H1N1',  'DEE5_H3N2',  'DUKE_HRV',  'UVA_HRV']

# Check testing samples avaiable or not. If not, dont perform prediction:
def checkTestingExperiments(_path):
    availableExperiments = []
    experiments = os.listdir(_path)
    experiments.sort()
    for folder in experiments:
        if os.path.exists(_path + "/" + folder + "/test"):
            availableExperiments.append(folder)

    return availableExperiments

def getOptimumParamsFS(expName,sc,timePoint,algorithm,fs_method,fs_wrapper,_mode):
    _filename = _mode + ".optimums"
    params = pd.read_csv("optimum_hyperparams/" + _filename,delimiter=";").astype(str)
    optimal = params.loc[(params['Experiment'] == expName) & (params['SubChallenge'] == "SC"+sc) & (params['TimePoint']==timePoint) & (params['Algorithm']==algorithm) & (params['Approach']==_mode) & (params['FeatureSelection']==fs_method) & (params['Wrapper']==fs_wrapper)]['Params'].values[0]

    return ast.literal_eval(optimal)

# Read optimum parameters from file according to Subchallenge/experiment/timepoint/algorithm and return pre-optimized hyper-parameters:
def getOptimumParams(expName,sc,timePoint,algorithm,_mode):
    _filename = _mode + ".optimums"
    params = pd.read_csv("optimum_hyperparams/" + _filename,delimiter=";").astype(str)
    optimal = params.loc[(params['Experiment'] == expName) & (params['SubChallenge'] == "SC"+sc) & (params['TimePoint']==timePoint) & (params['Algorithm']==algorithm) & (params['Approach']==_mode)]['Params'].values[0]

    return ast.literal_eval(optimal)


def getAverageOfProbabilities(_list):

    df_concat = pd.concat(_list)
    by_row_index = df_concat.groupby(df_concat.index)
    averageofProbabilities = by_row_index.mean()
    
    #print(by_row_index)

    return averageofProbabilities


def getPhaseTimePoints(expName,uptoTimePoint):

    #Find TimePoints in given phase (uptoTimePoint)
    timePoints = list(set([k.split(".")[2] for k in os.listdir(_datasetPath + "/" + expName + "/train") if int(k.split(".")[2]) <=int(uptoTimePoint) ]))
    return timePoints


def getTrainTestDataForVMApproach(expName,sc,timepoint):

    virusName = expName.split("_")[1]
    experimentsToBeMerged = [k for k in os.listdir(_datasetPath) if virusName in k]
    #experimentsToBeMerged.sort()

    _trainInputs,_trainSubjects,_trainTargets = [],[],[]

    if virusName == 'H3N2':
        #Since there are inconsistency among timepoitns of DEE2_H3N2 and DEE5_H3N2, time poitns to be merged should be specified:
        tpmapping = {'-30':'-24','2':'5','10':'12','18':'21'}
        for exp in experimentsToBeMerged:
            _traindataPath = _datasetPath + "/" + exp + "/train/"

            try:
                
                trainFileName = [k for k in os.listdir(_traindataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]
                trainFile = np.loadtxt(_traindataPath + trainFileName,dtype=str)

                _trainSubjects.append(trainFile[:,0])
                _trainInputs.append(trainFile[:,1:].astype(float))
                _trainTargets.append(pd.read_csv(_traindataPath + trainFileName + ".labels")["SC" +sc].tolist())

            except:

                trainFileName = [k for k in os.listdir(_traindataPath) if int(k.split(".")[2]) == int(tpmapping[timepoint]) and not k.endswith("labels")][0]
                trainFile = np.loadtxt(_traindataPath + trainFileName,dtype=str)

                _trainSubjects.append(trainFile[:,0])
                _trainInputs.append(trainFile[:,1:].astype(float))
                _trainTargets.append(pd.read_csv(_traindataPath + trainFileName + ".labels")["SC" +sc].tolist())

        pass
    else:
        
        for exp in experimentsToBeMerged:
            _traindataPath = _datasetPath + "/" + exp + "/train/"
            
            trainFileName = [k for k in os.listdir(_traindataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]
            trainFile = np.loadtxt(_traindataPath + trainFileName,dtype=str)

            _trainSubjects.append(trainFile[:,0])
            _trainInputs.append(trainFile[:,1:].astype(float))
            _trainTargets.append(pd.read_csv(_traindataPath + trainFileName + ".labels")["SC" +sc].tolist())

    
    trainSubjects, trainInputs, trainTargets = np.concatenate(_trainSubjects),np.concatenate(_trainInputs),np.concatenate(_trainTargets)

    # Use only test samples of given experiment. Test samples will not be merged from different experiments
    _testdataPath = _datasetPath + "/" + expName + "/test/"
    testFileName = [k for k in os.listdir(_testdataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]

    testFile = np.loadtxt(_testdataPath + testFileName,dtype=str)
    testSubjects, testInputs = testFile[:,0],testFile[:,1:].astype(float)
    testTargets = pd.read_csv(_testdataPath + testFileName + ".labels")["SC" +sc].tolist()

    return trainSubjects, trainInputs, trainTargets, testSubjects, testInputs, testTargets


def getTrainTestData(expName,sc,timepoint):

    _traindataPath = _datasetPath + "/" + expName + "/train/"
    trainFileName = [k for k in os.listdir(_traindataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]
    
    trainFile = np.loadtxt(_traindataPath + trainFileName,dtype=str)
    trainSubjects, trainInputs = trainFile[:,0],trainFile[:,1:].astype(float)
    trainTargets = pd.read_csv(_traindataPath + trainFileName + ".labels")["SC" +sc].tolist()


    _testdataPath = _datasetPath + "/" + expName + "/test/"
    testFileName = [k for k in os.listdir(_testdataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]

    testFile = np.loadtxt(_testdataPath + testFileName,dtype=str)
    testSubjects, testInputs = testFile[:,0],testFile[:,1:].astype(float)
    testTargets = pd.read_csv(_testdataPath + testFileName + ".labels")["SC" +sc].tolist()


    return trainSubjects, trainInputs, trainTargets, testSubjects, testInputs, testTargets


def applySelectedFeatures(trainInputs,testInputs,expName,sc,timePoint,_mode,fs_method,fs_wrapper):

    _filename = _mode + ".selectedfeatures"
    params = pd.read_csv("selected_features/" + _filename,delimiter=";").astype(str)
    optimal = params.loc[(params['Experiment'] == expName) & (params['SubChallenge'] == "SC"+sc) & (params['TimePoint']==timePoint) & (params['FSMethod']==fs_method) & (params['Wrapper']==fs_wrapper) & (params['Approach']==_mode)]['SelectedFetures'].values[0]

    selectedfeatures = ast.literal_eval(optimal)

    _indicies = np.loadtxt('datasets/Features.csv',dtype=str)

    sorter = np.argsort(_indicies)
    selectedFeaturesIndicies = sorter[np.searchsorted(_indicies, selectedfeatures, sorter=sorter)]

    return trainInputs[:,selectedFeaturesIndicies],testInputs[:,selectedFeaturesIndicies]


    #return trainInputs,testInputs

# Depending on Input argument use algorithms:
def getAlgorithm(algorithm):
    _algorithms = {"LR":LR(),"SVM":SVC(probability=True),"RF":RF(),"KNN":KNN(),"XGB":XGB(),"Lasso":Lasso(),"ElasticNet":ElasticNet(),"LinearSVR":LinearSVR(),"KNNR":KNeighborsRegressor(),"BayesRidge":BayesianRidge(),"Ridge":Ridge(),"GradientR":GradientBoostingRegressor(),"DTreeR":DecisionTreeRegressor(),"XGBR":XGBRegressor()}

    return _algorithms[algorithm]

# Inputs for Script:
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    # Input of sub-challenge type
    parser.add_argument('--SC', required=True, default='1', choices=['1','2','3'], help='Type of Sub-Challenge. IF SC-3 selected, classifier should be Regressor.')

    # Input of algoritm for prediction
    parser.add_argument('--algorithm', required=True, choices=['LR', 'SVM', 'KNN','RF', 'XGB','Lasso','ElasticNet','LinearSVR','KNNR','BayesRidge','Ridge','GradientR','DTreeR','XGBR'],
         help='Classification and Regression Algorithms to be used. LR, SVM, KNN, RF, XGB should be selected for SC-1 and SC-2, others are Regressor for SC-3 ')
    
    # Input of phase (Phase 1: T0, Phase 3: T24)
    parser.add_argument('--uptoTimePoint',
                        help='prediction up to T.0 hour or T.24 hour (i.e. Phase 1 or Phase)', required=True, default='0' , choices=['0', '24'])

    # Use pre-calculated optimum hyper-parameter space for algorithms.
    parser.add_argument('--useHyperParameters',
                        help='If True, models will uses hyper-parameters optimized before where stored in ".optimum" files', required=False, default='False' , choices=['True','False'])

    # Optimize parameters, (! takes long time)
    parser.add_argument('--optimizeHyperParameters',
                        help='If True, hyper-parameters of given algorithm will be optimized', required=False, default='False' , choices=['True','False'])


    parser.add_argument('--useVM',
                        help='If True, samples from different experiment but same viruses will be merged for extend number of traning samples (e.g. merge HRV DUKE and HRV UVA to increase number of samples in traning )'
                        , required=False, default='False' , choices=['True','False'])

    parser.add_argument('--useSelectedFeatures',
                        help='If True, Selected Features will be applied to train&test samples. Selected features are listed in selected_features folder according to approach (STPE or AF. VM features not available.). If use selected features, please specify fs_method and fs_wrapper args.'
                        , required=False, default='False' , choices=['True','False'])

    parser.add_argument('--fs_method',
                    help='Feature selection method. Please choice fisher_score,f_score,chi_square,gini_index,reliefF or mRMR for SC-1 and SC-2. For SC-3 mutual_info_regression or f_regression should be selected.'
                    , required=False, default='False' , choices=['fisher_score','f_score','chi_square','gini_index','reliefF','mRMR','mutual_info_regression','f_regression'])

    parser.add_argument('--fs_wrapper',
                    help='Wrapper algorithm for Feature selection method. Please choice LR,KNN or XGB for SC-1 and SC-2. For SC-3 Lasso, ElasticNet or GradientR'
                    , required=False, default='False' , choices=['LR','KNN','XGB','Lasso','ElasticNet','GradientR'])

    #parser.add_argument('--experiment',help='prediction up to T.0 hour or T.24 hour (i.e. Phase 1 or Phase)', required=False, default='ALL' , choices=experimentList)

    #return inputs
    args = parser.parse_args()
    return args


def main(args):

    _mode = "STPE"

    if args.useVM == 'True':
        _mode =_mode + "_VM"
    if args.useSelectedFeatures:
        _mode =_mode + "_FS"

    # Get Algorithm
    model = getAlgorithm(args.algorithm)
    sc = args.SC
    uptoTimePoint = args.uptoTimePoint
    useHyperParams = args.useHyperParameters

    # Get Experiments where test samples are avaiable.
    experiments = checkTestingExperiments(_datasetPath)
    experiments.sort()

    # GINI INDEX, mRMR and CHI-SQUARE features and optimum parameter will be updated
    #
    if args.fs_method in ['gini_index','reliefF','mRMR']:
        raise Exception("GINI INDEX, mRMR and CHI-SQUARE features and optimum parameter will be updated.")

    print("Avaiable testing samples for only {}".format(', '.join(experiments)))
    print()
    print(_mode)
    print("PARAMETERS: ")
    for key, value in vars(args).items():
        if value != 'False':
            print(' --',key,':',value)
    print()

    _list = []
    for expName in experiments:
        
        timePoints = getPhaseTimePoints(expName,uptoTimePoint)
        
        #print("TimePoints :",timePoints)
        
        for timePoint in timePoints:

            if args.useVM == 'True':
                trainSubjects, trainInputs, trainTargets, testSubjects, testInputs, testTargets = getTrainTestDataForVMApproach(expName,sc,timePoint)
            else:
                trainSubjects, trainInputs, trainTargets, testSubjects, testInputs, testTargets = getTrainTestData(expName,sc,timePoint)

            if args.useSelectedFeatures == 'True':
                trainInputs,testInputs = applySelectedFeatures(trainInputs,testInputs,expName,sc,timePoint,_mode,args.fs_method,args.fs_wrapper)

            # Parameter Optimization phase.
            if args.optimizeHyperParameters == 'True':
                if sc == '3':
                    _op = OP_regression(args.algorithm,model,trainInputs, trainTargets)
                else:
                    _op = OP_classification(args.algorithm,model,trainInputs, trainTargets)
                model.set_params(**_op.params)
                            

            if useHyperParams == 'True':
                if args.useSelectedFeatures == 'True':
                    _optimumparams = getOptimumParamsFS(expName,sc,timePoint,args.algorithm,args.fs_method,args.fs_wrapper,_mode)
                else:
                    _optimumparams = getOptimumParams(expName,sc,timePoint,args.algorithm,_mode)

                model.set_params(**_optimumparams)

            model.fit(trainInputs,trainTargets)

            if sc == "3":

                _testproba = model.predict(testInputs)
            
            else:
                
                _testproba = model.predict_proba(testInputs)

            _list.append(pd.DataFrame(data=np.c_[_testproba,testTargets],index=testSubjects))


    averageProbabilities = getAverageOfProbabilities(_list)
    print("PREDICTIONS: ")

    if args.SC == "3":

        averageProbabilities.columns = ["Prediction","Actual_Label"]
        predictions = averageProbabilities["Prediction"].to_numpy()
        actual_labels = averageProbabilities["Actual_Label"].to_numpy()
      
        pscore = pearsonr(actual_labels,predictions)[0]
        mse = mean_squared_error(actual_labels,predictions)

        print(averageProbabilities)
        print()
        print("Pearson : ",pscore)
        print("MSE",mse)
        

    else:

        averageProbabilities.columns = ["0","1","Actual_Label"]
        predictions = averageProbabilities[["0","1"]].to_numpy()
        actual_labels = averageProbabilities["Actual_Label"].to_numpy()
        
        precision, recall, thresholds = precision_recall_curve(actual_labels, predictions[:, 1])
        skauprc = auc(recall, precision)
        skauroc = roc_auc_score(actual_labels, predictions[:, 1])
        print(averageProbabilities)
        print()
        print("AUPRC :",skauprc)
        print("AUROC :",skauroc)



def more_main():
    args = parse_args()
    main(parse_args())

if __name__ == "__main__":
    more_main()
