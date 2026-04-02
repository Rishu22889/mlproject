import os
import sys

from dataclasses import dataclass
from urllib.parse import urlparse

import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier 

from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.utils import save_object, evaluate_models

import dagshub

dagshub.init(repo_owner='Rishu22889', repo_name='mlproject', mlflow=True)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing input data....")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'KNN': KNeighborsClassifier(),
                'SVM': SVC(),
                'Decision Tree': DecisionTreeClassifier(),
                'CatBoost': CatBoostClassifier(verbose=0),
                'XGBoost': XGBClassifier(eval_metric='logloss')
            }

            params = {
                'Logistic Regression': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear']
                },
                'Random Forest': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1]
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },
                'Decision Tree': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'CatBoost': {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [3, 5]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'colsample_bytree': [0.3, 0.7],
                    'subsample': [0.5, 1]
                }
            }

            model_report: dict = evaluate_models(models=models, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, param=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print("Best Model Found: ", best_model_name)

            model_names = list(params.keys())

            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model += model
            
            best_params = params[actual_model]

            # MlFlow
            mlflow.set_registry_uri("https://dagshub.com/Rishu22889/mlproject.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name="Model Training"):
                predicted_qualities = best_model.predict(X_test)

                (accuracy, precision, recall, f1) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_param("best_params", best_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score<0.6:
                raise CustomException("No suitable model found")
            logging.info(f"Best model found on both training and testing dataset is {best_model_name} with F1 score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            predicted = best_model.predict(X_test)
            final_score = f1_score(y_test, predicted)
            logging.info(f"Final F1 score for the best model is {final_score}")
            return final_score
            
        except Exception as e:
            raise CustomException(e, sys)