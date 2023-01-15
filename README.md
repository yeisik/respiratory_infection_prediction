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
- - 1 for prediction of individuals showing viral shedding, i.e., whether individual is infected or not.



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


