import os
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
import sys

def save_model(file_path, obj):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        with open(file_path, "wb") as obj_file:
            dill.dump(obj, obj_file)

    except Exception as e:
        raise CustomException(e, sys)
    
def model_eval(x_train, y_train, x_test, y_test, models, params):

    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i] 
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(x_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            r2_scores = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = r2_scores

            return report, gs.best_params_
        
    except Exception as e:
        raise CustomException(e, sys)
