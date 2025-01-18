import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from mlflow.models import infer_signature 
from urllib.parse import urlparse
from dataclasses import dataclass
import mlflow
import dagshub

import warnings
warnings.filterwarnings("ignore")



dagshub.init(repo_owner='ankitsharma5911', repo_name='MushroomClassification', mlflow=True)
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {    
                'Logistic Regression' : LogisticRegression(),
                'Decision Tree Regressor' : DecisionTreeClassifier(),
                "Naib Bias" : GaussianNB(),
                'SVC' : SVC(),
                'KNN': KNeighborsClassifier(),
                'Random Forest Classifier' : RandomForestClassifier(),
                'Adaboosting' : AdaBoostClassifier(),
                'GradientBoosting' : GradientBoostingClassifier()
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            y_pred = best_model.fit(X_train,y_train)
            precision_score = precision_score(y_test,y_pred)
            recall_score = recall_score(y_test,y_pred)
            f1_score = f1_score(y_test,y_pred)
            
            logging.log(f'Precision Score : {precision_score} , Recall Score : {recall_score} , F1 Score : {f1_score}')
            
            print(f'Precision Score : {precision_score} , Recall Score : {recall_score} , F1 Score : {f1_score}')
            
            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            signature = infer_signature(X_train,y_train)
            
            with mlflow.start_run():
                # mlflow.log_param("model", "Decision Tree Regressor")
                mlflow.log_metric("Accuracy",best_model_score)
                mlflow.log_metric("precision",precision_score)
                mlflow.log_metric("recall_score",recall_score)
                mlflow.log_metric("f1_score",f1_score)
                
                
                mlflow.set_experiment("MushroomClassification")
                
                mlflow.sklearn.log_model(
                                    sk_model=best_model,
                                    artifact_path="model",
                                    signature=signature,
                                    registered_model_name="register model")
                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    
                    mlflow.sklearn.log_model(best_model, "model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")
                
                
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            print('\n====================================================================================\n')
            
            logging.info(f'Best Model Found , Model Name : {best_model_name} , accuracy Score : {best_model_score}')
            
            print('Model Training Completed')
            logging.info('Model Training Completed')
            
        except Exception as e:
            CustomException(e,sys)