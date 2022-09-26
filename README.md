# Fraud Detection Example: Project Overview

**Tags: logistic regression, decision tree, random forest, hyperparameter tuning, SMOTE, cross validation, EDA, feature engineering, finance, fraud**

This notebook is part of Alura's course Modelos preditivos em dados: detecção de fraude (Data predictive models: fraud detection) by Sthefanie Monica ([Link](https://cursos.alura.com.br/course/modelos-preditivos-dados-deteccao-fraude)).

- Did the EDA to generate insights.
- Encoded categorical variables to fit in the model.
- Created two new columns to get a better picture of the data.
- Used SMOTE technique to balance the data.
- Built a logistic regression model as the base model. 
- Tested decision trees and random forests with RandomizeSearchCV to boost the hyperparameter.
- Draw recommendations to the security team.

## Code and resources

Platform: Jupyter Notebook

Python version: 3.7.6

Packages: imblearn, matplotlib, pandas, pandas profiling, numpy, seaborn and sklearn

## Data set

This is a Kaggle dataset called Fraud Detection Example and it has a fraction from [PaySim](https://github.com/EdgarLopezPhD/PaySim), which is a financial data simulator built for fraud detection purposes that emulate real-world data, generating synthetic records.

**Data set URL: https://www.kaggle.com/datasets/gopalmahadevan/fraud-detection-example**

## Model building

- Used SMOTE technique to balance the data, which is an oversampling technique that emulates new records using the KNN algorithm.
- Built a logistic regression (LR) model as the base model and check its score using ROC e AUC.
- Built a decision tree and random forest algorithm and compared it to LR.
- Boosted random forest hyperparameters with RandomizeSearchCV to enhance its performance. The parameters booster were:
    - n_estimators;
    - min_samples_split;
    - min_samples_leaf;
    - max_features;
    - max_depth;
    - criterion;
    - bootstrap.

## Model performance

The best parameters were:
- n_estimators': 100;
- min_samples_split: 8;
- min_samples_leaf: 4;
- max_features: log2;
- max_depth: 9;
- criterion: entropy;
- bootstrap: True.

The AUC metric for this model is 0.9996.
