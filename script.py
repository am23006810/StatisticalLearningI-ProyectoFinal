# Proyecto final
# Andrea Monzon 23006810
# Juan Pablo Rodas

import pandas as pd
import numpy as np

import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.transformation import LogTransformer

from feature_engine.outliers import OutlierTrimmer


def Train_model():
    print("train")
    dataTrain = pd.read_csv('train.csv')
    dataTest = pd.read_csv('test.csv')
    
    NUMERICAL_VARS_WITH_NA = ['Arrival Delay in Minutes']
    CATEGORICAL_VARIABLES = ['Gender', 'Customer Type', 'Type of Travel']
    CATEGORICAL_VARS_MAP = ['Class']
    OUTLIER = ['Arrival Delay in Minutes']
    
    pipeline_fe = Pipeline([
        #======== IMPUTACIONES =============

        # 1. Imputacion para variables numericas
        ('median_imputation', 
            MeanMedianImputer(imputation_method='median', variables=NUMERICAL_VARS_WITH_NA)),

        # 2. Codificacion de variables categoricas
        ('one_hot_encoding', 
           OneHotEncoder(variables=CATEGORICAL_VARIABLES)),

        # 3. Mapper
        ('categorical_encoder',
            OrdinalEncoder(encoding_method='ordered', variables=CATEGORICAL_VARS_MAP)
        ),

        # 4. Outlier
        ('outlier',
            OutlierTrimmer(capping_method='iqr', tail='right', fold=1.5, variables=OUTLIER)
        ),

        # 5. Feature Scaling
        ('scaler',
            MinMaxScaler()),
        
        ('modelo_lasso', 
             Lasso(alpha=0.01, random_state=2022)
        )
    ])
    
    

    
def predict():
    