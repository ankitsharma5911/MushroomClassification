import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass
    def predict(self,features):
        try:
            # load the model
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)   

class CustomData:
    def __init__(self,
                 cap_shape:str,
                 cap_surface:str,
                 cap_color:str,
                 bruises:str,
                 odor:str,
                 gill_attachment:str,
                 gill_spacing:str,
                 gill_size:str,
                 gill_color:str,
                 stalk_shape:str,
                 stalk_root:str,
                 stalk_surface_above_ring:str,
                 stalk_surface_below_ring:str,
                 stalk_color_above_ring:str,
                 stalk_color_below_ring:str,
                 veil_type:str,
                 veil_color:str,
                 ring_number:str,
                 ring_type:str,
                 spore_print_color:str,
                 population:str,
                 habitat:str
                 ) :
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'cap-shape':[self.cap_shape],
                'cap-surface':[self.cap_surface],
                'cap-color':[self.cap_color],
                'bruises':[self.bruises],
                'odor':[self.odor],
                'gill-attachment':[self.gill_attachment],
                'gill-spacing':[self.gill_spacing],
                'gill-size':[self.gill_size],
                'gill-color':[self.gill_color],
                'stalk-shape':[self.stalk_shape],
                'stalk-root':[self.stalk_root],
                'stalk-surface-above-ring':[self.stalk_surface_above_ring],
                'stalk-surface-below-ring':[self.stalk_surface_below_ring],
                'stalk-color-above-ring':[self.stalk_color_above_ring],
                'stalk-color-below-ring':[self.stalk_color_below_ring],
                'veil-type':[self.veil_type],
                'veil-color':[self.veil_color],
                'ring-number':[self.ring_number],
                'ring-type':[self.ring_type],
                'spore-print-color':[self.spore_print_color],
                'population':[self.population],
                'habitat':[self.habitat]

            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            logging.info(df.head(5))
            return df
        except Exception as e:
            CustomException(e,sys)
