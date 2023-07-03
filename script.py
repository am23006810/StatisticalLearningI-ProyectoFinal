# Proyecto final
# Andrea Monzon 23006810
# Juan Pablo Rodas

import pandas as pd
import numpy as np

import datetime
import time

import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.transformation import LogTransformer

from feature_engine.outliers import OutlierTrimmer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def Train_model():
    print("train")
    dataTrain = pd.read_csv('train.csv')
    dataTest = pd.read_csv('test.csv')
    
    var_preds = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Gender', 'Customer Type', 'Type of Travel', 'Class']
    
    y_test = dataTest['satisfaction']
    y_train = dataTrain['satisfaction']
    
    y_train = pd.get_dummies(y_train,drop_first=True)
    y_test = pd.get_dummies(y_test,drop_first=True)
    
    NUMERICAL_VARS_WITH_NA = ['Arrival Delay in Minutes']
    CATEGORICAL_VARIABLES = ['Gender', 'Customer Type', 'Type of Travel']
    CATEGORICAL_VARS_MAP = ['Class']
    OUTLIER = ['Arrival Delay in Minutes']
    
    
    fecha_hora_inicio = datetime.datetime.now()
    
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

        # 3. Feature Scaling
        ('scaler',
            MinMaxScaler()),

        ('modelo', 
             RandomForestClassifier(criterion = 'entropy', max_depth=None, n_estimators= 150)
        )
    ])
    
    pipeline_fe.fit(dataTrain[var_preds], y_train['satisfied'])
    
    fecha_hora_fin = datetime.datetime.now()
    
    
    predictions = pipeline_fe.predict(dataTest[var_preds])
    accuracy = accuracy_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    roc_auc = roc_auc_score(y_test, predictions)
    
    
    tiempo_transcurrido = fecha_hora_fin - fecha_hora_inicio
    tiempo_transcurrido_segundos = tiempo_transcurrido.total_seconds()
    
    nombre_archivo = fecha_hora_inicio.strftime('%Y%m%d_%H%M%S') + '.txt'
    
    with open(nombre_archivo, 'a') as archivo:
        archivo.write('Fecha y hora de ejecución: {}\n'.format(fecha_hora_inicio))
        archivo.write('Tiempo transcurrido (segundos): {}\n'.format(tiempo_transcurrido_segundos))
        archivo.write('Accuracy: {}\n'.format(accuracy))
        archivo.write('Specificity: {}\n'.format(specificity))
        archivo.write('Sensitivity: {}\n'.format(sensitivity))
        archivo.write('\n')
    
    
def predict():
    print(predict)
    
    
    
if __name__ == '__main__':
    Train_model()