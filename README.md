# Scripts of STPE & AF Approaches for Respiratory Infection Prediction

## Usage of Scripts

```bash
python run_STPE.py --SC 1 --algorithm LR --uptoTimePoint 0
```

```bash
python run_AF.py --SC 1 --algorithm LR --uptoTimePoint 0
```

#### Required Options
- --SC, Subchallenges of Respiratory DREAM Viral Challenge. 
- - "1", Prediction of individuals showing viral shedding, i.e., whether individual is infected or not. Binary outcome.
- - "2", Predict symptomatic response to exposure. Binary outcome.
- - "3", Continuous value prediction of symptom score after exposure. Continous outcome.

- --uptoTimePoint, The time point to be predicted. 
- - 0 for Phase 1
- - 24 for Phase 3
- - Other time points are also supported. (i.e. if set "4", than models will be predicted up to 4.th hour).

- --algorithm, Classification and Regression algorithms to be used. 
- - 'LR', 'SVM', 'KNN', 'RF', 'XGB' should be selected for SC-1 and SC-2.
- - 'Lasso','ElasticNet','LinearSVR','KNNR','BayesRidge','Ridge','GradientR','DTreeR','XGBR' are Regressor for SC-3 . 

#### Other Options
- --useVM, VirusMerge option. If True, samples from different experiment but same viruses will be merged to extend number of traning data.
-  --useSelectedFeatures, If True, models will use selected features for each experiment and timepoint. Selected features are listed in selected_features folder according to approach (STPE or AF. VM features not available.). If use selected features, please specify fs_method and fs_wrapper parameters. Note that, this parameters not apply "feature selection", just to use apply pre-selected features.
- - --fs_method, Feature Selection Method,  
- - - Please choice fisher_score,f_score,chi_square,gini_index,reliefF or mRMR for SC-1 and SC-2. 
- - - For SC-3 mutual_info_regression or f_regression should be selected.
- - --fs_wrapper Wrapper algorithm of feature selection. 
- - - Please choice LR,KNN or XGB for SC-1 and SC-2. 
- - - For SC-3 Lasso, ElasticNet or GradientR should be selected.

- --useHyperParameters, If True, models will use pre-optimized hyper-parameters for each algorithm/experiment and timepoint. Hyperparameters are stored in ".optimum" files under the folder "optimum_hyperparams". According approach (AF,STPE, AF_VM, STPE_VM, AF_FS or AF_STPE) optimum hyper-parameters will be applied automatically during training process. This options not apply "hyper-parameter optimization". Its only for to use pre-optimized parameters. To apply hyper-parameter optimization from 	scratch, plaese use `--useHyperParameters` option.

- --useHyperParameters, If True, models will apply hyper-parameter optimization before the training process on traning data using `paramaterOptimizerClassification.py` and `paramaterOptimizerRegression.py` scripts. This options not apply "hyper-parameter optimization".  



