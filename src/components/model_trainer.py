import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import model_eval, save_model

@dataclass
class ModelTrainerConfig:
    model_train_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, x_train, x_test):
        try:
            logging.info('Seperating dataset into input and output')
            x_train, y_train, x_test ,y_test = (
                x_train[:, :-1],
                x_train[:, -1],
                x_test[:, :-1],
                x_test[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            
            }

        
            model_report, _= model_eval(x_train, y_train, x_test, y_test, models=models, params=params) 

            best_score = max(sorted(list(model_report.values())))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_score)
                ]
            
            best_model = models[best_model_name]

            if best_score<0.6:
                logging.info('No best model found')
            logging.info('Best model for training and testing data')

            logging.info('Saving best model')

            save_model(
                file_path=self.model_trainer_config.model_train_path,
                obj=best_model
            )

            logging.info('Making predictions on model and calculating r2_score')
            y_pred = best_model.predict(x_test)
            r2_scores = r2_score(y_test, y_pred)

            return r2_scores, best_model_name, _

        except Exception as e:
            raise CustomException(e, sys)