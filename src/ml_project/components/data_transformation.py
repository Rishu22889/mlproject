import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
from src.ml_project.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logging.info("Dropping unnecessary columns")
        return X.drop(self.columns, axis=1)


class TitleExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            logging.info("Extracting titles from Name column")
            X = X.copy()

            X['Title'] = X['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

            X['Title'] = X['Title'].replace([
                'Lady', 'Countess', 'Capt', 'Col',
                'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'
            ], 'Rare')

            X['Title'] = X['Title'].replace({
                'Mlle': 'Miss',
                'Ms': 'Miss',
                'Mme': 'Mrs'
            })

            X = X.drop('Name', axis=1)
            return X

        except Exception as e:
            raise CustomException(e, sys)


class FamilySizeCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            logging.info("Creating FamilySize feature")
            X = X.copy()
            X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
            return X

        except Exception as e:
            raise CustomException(e, sys)


class LogFare(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            logging.info("Applying log transformation to Fare")
            X = X.copy()
            X['Fare'] = np.log(X['Fare'] + 1)
            return X

        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating preprocessing pipeline")

            numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'FamilySize']
            categorical_features = ['Sex', 'Embarked', 'Title']

            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first'))
            ])

            preprocessor = ColumnTransformer([
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ])

            data_pipeline = Pipeline([
                ('drop_cols', DropColumns(['PassengerId', 'Ticket', 'Cabin'])),
                ('title', TitleExtractor()),
                ('family', FamilySizeCreator()),
                ('log_fare', LogFare()),
                ('preprocessor', preprocessor)
            ])

            return data_pipeline

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop('Survived', axis=1)
            y_train = train_df['Survived']

            X_test = test_df.drop('Survived', axis=1)
            y_test = test_df['Survived']

            pipeline = self.get_data_transformer_object()

            logging.info("Fitting pipeline on training data")
            X_train_transformed = pipeline.fit_transform(X_train)

            logging.info("Transforming test data")
            X_test_transformed = pipeline.transform(X_test)

            # Combine with target
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save pipeline
            save_object(self.data_transformation_config.preprocessor_obj_file_path, pipeline)

            logging.info("Pipeline saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)