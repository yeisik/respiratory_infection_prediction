from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

# NUMPY VERSION '1.22.4'
import numpy as np
import pandas as pd
import json
import ast
import sys

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

_datasetPath = "datasets/"
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
def getOptimumParams(expName,sc,uptoTimePoint,algorithm,_mode):
    _filename = _mode + ".optimums"
    params = pd.read_csv("optimum_hyperparams/" + _filename,delimiter=";").astype(str)
    
    optimal = params.loc[(params['Experiment'] == expName) & (params['SubChallenge'] == "SC"+sc) & (params['TimePoint']==uptoTimePoint) & (params['Algorithm']==algorithm) & (params['Approach']==_mode)]['Params'].values[0]

    return ast.literal_eval(optimal)


def getPhaseTimePoints(expName,uptoTimePoint):

    #Find TimePoints in given phase (uptoTimePoint)
    timePoints = list(set([k.split(".")[2] for k in os.listdir(_datasetPath + "/" + expName + "/train") if int(k.split(".")[2]) <=int(uptoTimePoint) ]))
    return timePoints


def getTrainTestDataForVMApproach(expName,sc,uptoTimePoint):

    virusName = expName.split("_")[1]
    experimentsToBeMerged = [k for k in os.listdir(_datasetPath) if virusName in k]
    
    _trainlist = []
    _testlist = []
    
    for experiment in experimentsToBeMerged:
        timepoints = getPhaseTimePoints(experiment,uptoTimePoint)

        for timepoint in timepoints:

            _traindataPath = _datasetPath + "/" + experiment + "/train/"        
            trainFileName = [k for k in os.listdir(_traindataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]

            trainData = pd.read_csv(_traindataPath + trainFileName,sep=" ",header=None,index_col=0)
            trainData["CLASSLABEL"] = pd.read_csv(_traindataPath + trainFileName + ".labels")["SC" +sc].tolist()
            _trainlist.append(trainData)

            try:
                _testdataPath = _datasetPath + "/" + experiment + "/test/"
                testFileName = [k for k in os.listdir(_testdataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]

                testData = pd.read_csv(_testdataPath + testFileName,sep=" ",header=None,index_col=0)
                testData["CLASSLABEL"] = pd.read_csv(_testdataPath + testFileName + ".labels")["SC" +sc].tolist()
                _testlist.append(testData)
            except Exception as e:
                pass
        

    averageTrain = pd.concat(_trainlist)
    averageOfFeaturesTrain = averageTrain.groupby(averageTrain.index).mean()
    trainTargets = averageOfFeaturesTrain["CLASSLABEL"].to_numpy()
    trainSubjects,trainInputs = averageOfFeaturesTrain.index.tolist(),averageOfFeaturesTrain.to_numpy()[:,:-1]

    averageTest = pd.concat(_testlist)
    averageOfFeaturesTest = averageTest.groupby(averageTest.index).mean()
    testTargets = averageOfFeaturesTest["CLASSLABEL"].to_numpy()
    testSubjects,testInputs = averageOfFeaturesTest.index.tolist(),averageOfFeaturesTest.to_numpy()[:,:-1]
        
    return trainSubjects, trainInputs, trainTargets, testSubjects, testInputs, testTargets


def getTrainTestData(expName,sc,uptoTimePoint):

    timepoints = getPhaseTimePoints(expName,uptoTimePoint)

    _trainlist = []
    _testlist = []


    for timepoint in timepoints:

        _traindataPath = _datasetPath + "/" + expName + "/train/"        
        trainFileName = [k for k in os.listdir(_traindataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]

        trainData = pd.read_csv(_traindataPath + trainFileName,sep=" ",header=None,index_col=0)
        trainData["CLASSLABEL"] = pd.read_csv(_traindataPath + trainFileName + ".labels")["SC" +sc].tolist()

        _testdataPath = _datasetPath + "/" + expName + "/test/"
        testFileName = [k for k in os.listdir(_testdataPath) if int(k.split(".")[2]) == int(timepoint) and not k.endswith("labels")][0]

        testData = pd.read_csv(_testdataPath + testFileName,sep=" ",header=None,index_col=0)
        testData["CLASSLABEL"] = pd.read_csv(_testdataPath + testFileName + ".labels")["SC" +sc].tolist()

        _trainlist.append(trainData)
        _testlist.append(testData)

    averageTrain = pd.concat(_trainlist)
    averageOfFeaturesTrain = averageTrain.groupby(averageTrain.index).mean()


    trainTargets = averageOfFeaturesTrain["CLASSLABEL"].to_numpy()
    trainSubjects,trainInputs = averageOfFeaturesTrain.index.tolist(),averageOfFeaturesTrain.to_numpy()[:,:-1]

    averageTest = pd.concat(_testlist)
    averageOfFeaturesTest = averageTest.groupby(averageTest.index).mean()

    testTargets = averageOfFeaturesTest["CLASSLABEL"].to_numpy()
    testSubjects,testInputs = averageOfFeaturesTest.index.tolist(),averageOfFeaturesTest.to_numpy()[:,:-1]

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
    
    #parser.add_argument('--experiment',                        help='prediction up to T.0 hour or T.24 hour (i.e. Phase 1 or Phase)', required=False, default='ALL' , choices=experimentList)

    #return inputs
    args = parser.parse_args()
    return args

def main(args):
  # Get Algorithm
    model = getAlgorithm(args.algorithm)
    sc = args.SC
    uptoTimePoint = args.uptoTimePoint
    useHyperParams = args.useHyperParameters

    _mode = "AF"
    if args.useVM == 'True':
        _mode =_mode + "_VM"
    if args.useSelectedFeatures == 'True':
        _mode =_mode + "_FS"

    # Get Experiments where test samples are avaiable.
    experiments = checkTestingExperiments(_datasetPath)
    
    print("Avaiable testing samples for only {}".format(', '.join(experiments)))
    print()
    print(_mode)
    print("PARAMETERS: ")
    for key, value in vars(args).items():
        if value != 'False':
            print(' --',key,':',value)
    print()

    predictions = []
    actual_labels = []
    subjects = []

    for expName in experiments:
        
        if args.useVM == 'True':
            trainSubjects, trainInputs, trainTargets, testSubjects, testInputs, testTargets = getTrainTestDataForVMApproach(expName,sc,uptoTimePoint)
        else:
            trainSubjects, trainInputs, trainTargets, testSubjects, testInputs, testTargets = getTrainTestData(expName,sc,uptoTimePoint)

        if args.useSelectedFeatures == 'True':
            trainInputs,testInputs = applySelectedFeatures(trainInputs,testInputs,expName,sc,uptoTimePoint,_mode,args.fs_method,args.fs_wrapper)


        # Parameter Optimization phase.
        if args.optimizeHyperParameters == 'True':
            if sc == '3':
                _op = OP_regression(args.algorithm,model,trainInputs, trainTargets)
            else:
                _op = OP_classification(args.algorithm,model,trainInputs, trainTargets)
            model.set_params(**_op.params)
                            

        if useHyperParams == 'True':
            try:
                if args.useSelectedFeatures == 'True':
                    _optimumparams = getOptimumParamsFS(expName,sc,uptoTimePoint,args.algorithm,args.fs_method,args.fs_wrapper,_mode)
                else:
                    _optimumparams = getOptimumParams(expName,sc,uptoTimePoint,args.algorithm,_mode)
                
                model.set_params(**_optimumparams)
            except:
                raise("Optimum parameters not found.")
                
            #print(model)

        model.fit(trainInputs,trainTargets)
    
        if sc == "3":
            testproba = model.predict(testInputs)
            
        else:
            testproba = model.predict_proba(testInputs)

        predictions.append(testproba)
        actual_labels.append(testTargets)
        subjects.append(testSubjects)
    

    predictions,actual_labels,subjects = np.concatenate(predictions, axis=0),np.concatenate(actual_labels, axis=0),np.concatenate(subjects, axis=0)
    
    averageofFeatures = pd.DataFrame(data = np.c_[predictions,actual_labels],index=subjects)

    print("PREDICTIONS: ")

    if args.SC == "3":
        averageofFeatures.columns = ["Prediction","Actual_Label"]
        print(averageofFeatures)
        pscore = pearsonr(actual_labels,predictions)[0]
        mse = mean_squared_error(actual_labels,predictions)

        print()
        print("Pearson : ",pscore)
        print("MSE",mse)

    else:
        averageofFeatures.columns = ["0","1","Actual_Label"]
        print(averageofFeatures)
        precision, recall, thresholds = precision_recall_curve(actual_labels, predictions[:, 1])
        skauprc = auc(recall, precision)
        skauroc = roc_auc_score(actual_labels, predictions[:, 1])

       
        print()
        print("AUPRC :",skauprc)
        print("AUROC :",skauroc)
        

def more_main():
    args = parse_args()
    main(parse_args())

if __name__ == "__main__":
    more_main()
