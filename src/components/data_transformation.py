import os 
import sys
import numpy as np
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
# importing liberary for datapipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

## Data transformation config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class  DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self,columns):
        try:
            # loading preprocessor object from file
            logging.info("data transformation start.")

            cat_pipeline = Pipeline(
                steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalEncoding',OrdinalEncoder())]
            )
            preprocessor = ColumnTransformer([('cat_pipeline',cat_pipeline,columns)])

            return  preprocessor
            logging.info("pipeline competed")
        
        except Exception as e:
            logging.error(f"An error occurred in pipeline: {e}")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            # reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')


             # features into independent and dependent features
            X_train_df = train_df.iloc[:,1:]
            y_train_df = train_df.iloc[:,0].map({'e':1,'p':0})

            X_test_df = test_df.iloc[:,1:]
            y_test_df = test_df.iloc[:,0].map({'e':1,'p':0})

            columns =[col for col in  X_train_df.columns]

            preprocessor_obj = self.get_data_transformation_object(columns)

            # apply transformation
            X_train_df = preprocessor_obj.fit_transform(X_train_df)
            X_test_df = preprocessor_obj.transform(X_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            # logging.info("scalling data are",pd.DataFrame(X_train_df,columns=preprocessor_obj.get_feature_names_out()).head(3))

            train_arr = np.c_[X_train_df,np.array(y_train_df)]
            test_arr = np.c_[X_test_df,np.array(y_test_df)]

            file_path = self.data_transformation_config.preprocessor_obj_file_path

            obj = preprocessor_obj

            save_object(file_path=file_path,obj=obj)

            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,
                test_arr
            )



        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
    


