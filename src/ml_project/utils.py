import os
import sys
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


import pickle
import numpy as np

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established", mydb)

        df = pd.read_sql_query('select * from train', mydb)
        logging.info("Reading SQL database completed")
        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(models, X_train, y_train, X_test, y_test,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_train_prob = model.predict_proba(X_train)[:, 1]
                y_test_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_train_prob = None
                y_test_prob = None

            train_model_score = f1_score(y_train, y_train_pred)
            test_model_score =  f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)