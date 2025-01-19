import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


obj = DataIngestion()
train_data_path,test_data_path = obj.initialize_data_ingestion()
print(train_data_path,test_data_path)

data_trans = DataTransformation()


train_arr,test_arr = data_trans.initiate_data_transformation(train_data_path,test_data_path)
print(train_arr[:3],test_arr[:3])


model_training = ModelTrainer()
best_model_name = model_training.initate_model_training(train_arr,test_arr)

print(best_model_name)


