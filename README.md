
# PREDICT BIOLOGICAL CLOCK

## STRUCTURE

There are these folders: Data, Results, src and tests.

    - Data: this folder is empty in github but you have to move your xlsx to this folder

    - Results: 

        - Prediction and true patients in csv

        - Graphic with result

    - src has the following structure:
 
        - Ensemble: It is the main class in this project. This executes the steps from full process of ML
    
        - experimental_notebook: fast proofs for understanding the problem to resolve
    
        - FeatureSelection: It is the step where you reduce the features to get the most important features
     
        - Models: It is the folder where you execute models about predictions when the data is cleaned
    
        - Preprocessors: It is the folder with the scripts about clean and imput variables


    - tests: test for trying functions with base cases

    - proof_code.py:  script to test the whole flow and get results 


There are also:
    
    - .gitignore not to upload data files
    
    - .pytest.ini to test the tests
    
    - README.md to understand the structure
    
    - requirements.txt with the necessary libraries and their versions


## Step1: CLEAN DATA

### NECCESARY COLUMNS

Firstly, we choose the variables that we check if they are important, in this case, fitbit and proteins

Remove duplicate columns

Remove constants columns

### IMPUT NA VALUES

Imput na values with algorithms because there are a lot of variables but we have little cases, then we can use algorithms very quickly

We have several algorithms and we can choose the best of ['mean_mode', 'knn', 'linear', 'logistic', 'svm', 'xgboost', 'ensemble']

### FINAL STEPS

We convert dummies columns with non numerical data

We normalize

Finally, we have the clean dataset

## Step2: FEATURE SELECTION

### Remove columns with little variance

We can choose of threshold and we remove columns that they are almost constant

### Selection features with algorithms

We have 4 algorithms to get variables. 

Each algorithm gives us the most important variables and their importance. 

We put a threshold and get the variables more importance than that threshold. 

We remove the duplicated variables of that list to get a list with unique values


## Step3: ML MODELS TO PREDICT BIOLOGICAL CLOCK

As we have cleaned data and reduced data we can apply supervised models

I choose the following models:
    
    - Linear
    
    - XGBoost

    - LightGBM

    - Ensemble with the same importance


We choose the best hyperparameters and we can ensemble both of them


## Step4: METRICS, RMSE AND GRAPHICS

As it is a regression model, we choose rmse as metric and we can show results in a graphic too

Files are in:

    - Results/graphics.png

    - Results/PredictedVsTrue.csv


## RESULTS

Best result is in progress
