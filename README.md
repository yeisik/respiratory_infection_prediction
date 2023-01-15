# Scripts of STPE & AF Approaches for Respiratory Infection Prediction

## Usage of Scripts

```bash
python run_STPE.py --SC 1 --algorithm LR --uptoTimePoint 0
```

```bash
python run_AF.py --SC 1 --algorithm LR --uptoTimePoint 0
```

#### General Options
- --SC, Subchallenges of Respiratory DREAM Viral Challenge. 
- - "1", Prediction of individuals showing viral shedding, i.e., whether individual is infected or not. Binary outcome.
- - "2", Predict symptomatic response to exposure. Binary outcome.
- - "3", Continuous value prediction of symptom score after exposure. Continous outcome.

- --uptoTimePoint, The time point to be predicted. 0 for Phase 1 and 24 for Phase 3. Other time points are also supported.
- --algorithm, Classification and Regression algorithms to be used. 'LR', 'SVM', 'KNN', 'RF', 'XGB' should be selected for SC-1 and SC-2. 'Lasso','ElasticNet','LinearSVR','KNNR','BayesRidge','Ridge','GradientR','DTreeR','XGBR' are Regressor for SC-3 . 


- --input, input graph file. Only accepted edgelist format. 
- --output, output graph embedding file. 
- --task, choose to evaluate the embedding quality based on a specific prediction task (i.e., link-prediction, node-classification, none (no eval), default is none) 
- --testing-ratio, testing set ratio for prediction tasks. Only applied when --task is not none. The default is 0.2 
- --dimensions, the dimensions of embedding for each node. The default is 100. 
- --method, the name of embedding method 
- --label-file, the label file for node classification.  
- --weighted, true if the input graph is weighted. The default is False.
- --eval-result-file, the filename of eval result (save the evaluation result into a file). Skip it if there is no need.
- --seed, random seed. The default is 0. 


