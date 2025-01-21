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

@dataclass 
class ModelPreformance:
    accuracy: float = ""
    precision: float = ""
    recall: float = ""
    f1: float  = ""
    
    
class ModelTrainer:
    def __init__(self)->None:
        self.model_trainer_config = ModelTrainerConfig()
        self.performance_matrics = ModelPreformance()
    def register_model(self,model,X_train,y_train,modelperformance:ModelPreformance):
        try:
            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            signature = infer_signature(X_train,y_train)
            print(modelperformance.accuracy)
            
            
            with mlflow.start_run():
                mlflow.log_metric("Accuracy",modelperformance.accuracy)
                mlflow.log_metric("precision",modelperformance.precision)
                mlflow.log_metric("recall_score",modelperformance.recall)
                mlflow.log_metric("f1_score",modelperformance.f1)
                
                # mlflow.set_experiment("MushroomClassification")
                
                mlflow.sklearn.log_model(
                                    sk_model=model,
                                    artifact_path="artifacts",
                                    signature=signature,
                                    registered_model_name="best model")
            
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "artifacts",registered_model_name="best model")
                else:
                    mlflow.sklearn.log_model(model, "artifacts")
        except Exception as e:
            logging.error(f"An error occurred in registering model: {e}")
            CustomException(e,sys) 
 
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            logging.info('Model training started')
            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'GaussianNB': GaussianNB(),
                'SVC': SVC(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier()
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
            
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            
            
            
            return best_model
          
        except Exception as e:
            logging.error(f"An error occurred in training model: {e}")
            raise CustomException(e,sys)
                
    def save_model(self,model):
        try:
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            logging.info('Model saved successfully')
        except Exception as e:
            logging.error(f"An error occurred in saving model: {e}")
            raise CustomException(e,sys)
    
    def evaluate_model(self,X_test,y_test,model,model_performance:ModelPreformance):
        try:
            y_pred = model.predict(X_test)
            
            # model_performance = ModelPreformance()
            
            
            model_performance.accuracy = accuracy_score(y_test,y_pred)
            model_performance.precision = precision_score(y_test,y_pred)
            model_performance.recall = recall_score(y_test,y_pred)
            model_performance.f1 = f1_score(y_test,y_pred)
        
            return model_performance
        
        
        except Exception as e:
            logging.error(f"An error occurred in evaluating model: {e}")
            raise CustomException(e,sys)
        
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Train model
            model=self.train_model(X_train,y_train,X_test,y_test)
            logging.info('Model Training Completed')
            print('Model Training Completed')
            
            # Save model
            self.save_model(model)
            logging.info('Model saved successfully')
            print('Model saved successfully')
            
            
            # Evaluate model
            matrics:ModelPerformance = self.evaluate_model(X_test,y_test,model,self.performance_matrics)
            logging.info('Model evaluation completed')
            print('Model evaluation completed')
            
            
            # Register model
            self.register_model(model,X_train,y_train,matrics)
            logging.info('Model registered successfully')
            print('Model Training Completed')
            
            
            
        except Exception as e:
            logging.error(f"An error occurred in model training: {e}")
            raise CustomException(e,sys)