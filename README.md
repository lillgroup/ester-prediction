# Computational Prediction of De-Esterification by Human Carboxylesterases 1 and 2

Author : Rüthemann Peter, André Fischer

## Description

Computational Prediction of De-Esterification by Human Carboxylesterases 1 and 2 using Random Forests (RFs) trained on 
literature-derived dataset.

This repository contains all used structures as [sdf](compounds/SDF) or [mae](compounds/MAE) files, as well as the used [data set](input/datasets.csv) for training.
The predictions for the [training](prediction/TRAIN.csv), as well as the two test sets [EXT_A](prediction/EXT_A.csv) and [EXT_B](prediction/EXT_B.csv) are provided.

The two scripts `Train.py` and `Prediction.py` allow the reproduction of the results in table 1 and the generation of the Random Forest (RF) model.

```
Dataset:
    - input/datasets.csv:       Data sets with descriptor data and labels (y_true)

Random forest model
    - input/features.csv:       Selected features used for training
    - input/parameters.yaml:    Hyper parameters required for initialisation of sklearn Random Forest
    - model/RF_classifier.pkl:  Random forest model stored as pickle file

Results files:
    - prediction/TRAIN.csv:     Dataset with predictions for training set
    - prediction/EXT_A.csv:     Dataset with predictions for external test set A
    - prediction/EXT_B.csv:     Dataset with predictions for external test set B  
```


## SetUp

>1. [Install conda](https://docs.conda.io/en/latest/miniconda.html)
>3. Install conda environment \
> `conda env create -f environment.yml` 

## Training
> 1. Activate environment \
> `conda activate ester-prediction`
> 2. Execute training of training set \
> `python3 Train.py` 

## Prediction
> 1. Activate environment \
> `conda activate ester-prediction`
> 2. Execute prediction of training and two test sets \
> `python3 Prediction.py` 
