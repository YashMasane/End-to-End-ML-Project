from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_model
from dataclasses import dataclass

import os
import sys

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.datatransformerconfig = DataTransformationConfig()

    def get_data_preprocessing_obj(self):
        '''
        returns data transformation object which will do required
        data transformation
        '''

        try:
            num_columns = ["writing_score", "reading_score"]
            cat_columns = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course"
            ]

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoding', OneHotEncoder()),
                    ('standardscaler', StandardScaler(with_mean=False))
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('standardscaler', StandardScaler())
                ]
            )

            logging.info(f'Categorical features {cat_columns}')
            logging.info(f'Numerical features {num_columns}')

            pipeline = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ]
            )

            return pipeline
        except Exception as e:
            raise CustomException(e, sys)
        

    
    def initialte_data_transformation(self, train_path, test_path):
        '''
        returns transformed data
        '''
 
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_preprocessing_obj()

            target_variable = 'math_score'

            input_train_data = train_df.drop(target_variable, axis=1)
            output_train_data = train_df[target_variable]

            input_test_data = test_df.drop(target_variable, axis=1)
            output_test_data = test_df[target_variable]

            logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe."
                )


            input_train_features = preprocessor_obj.fit_transform(input_train_data)
            input_test_features = preprocessor_obj.transform(input_test_data)

            train_arr = np.c_[input_train_features, np.array(output_train_data)]
            test_arr = np.c_[input_test_features, np.array(output_test_data)]

            logging.info(f"Saving preprocessing object.")

            save_model(
                file_path = self.datatransformerconfig.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.datatransformerconfig.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

