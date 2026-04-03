import os
import sys
import pandas as pd
import numpy as np
import pickle

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging


class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model_path = os.path.join('artifacts', 'model.pkl')
        
        # Load model
        try:
            self.model = self._load_object(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logging.warning(f"Model not found at {self.model_path}: {e}")
            self.model = None
    
    def _load_object(self, file_path):
        try:
            with open(file_path, 'rb') as file_obj:
                obj = pickle.load(file_obj)
            logging.info(f"Loaded object from {file_path}")
            return obj
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, passenger_data):
        """
        Make prediction for a single passenger.
        
        Args:
            passenger_data: dict with keys: Pclass, Sex, Age, Fare, SibSp, Parch, Embarked
            
        Returns:
            tuple: (prediction, probability)
        """
        try:
            if self.model is None:
                raise CustomException("Model not loaded. Train and save model first.", sys)
            
            # Create dataframe from single passenger data
            df = pd.DataFrame([passenger_data])
            
            # Apply manual preprocessing (matching training data transformations)
            df = self._preprocess_features(df)
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            # Get prediction probability
            try:
                probability = self.model.predict_proba(df)[0][1]
            except:
                probability = 0.5  # Fallback if model doesn't support predict_proba
            
            logging.info(f"Prediction: {prediction}, Probability: {probability}")
            return prediction, probability
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _preprocess_features(self, df):
        """
        Apply manual preprocessing to match the training pipeline.
        This avoids the full sklearn pipeline which expects raw data with all columns.
        """
        try:
            df = df.copy()
            
            # 1. Handle missing Age with median (using training median as fallback)
            if 'Age' in df.columns:
                df['Age'].fillna(25.0, inplace=True)
            
            # 2. Create FamilySize feature
            if 'SibSp' in df.columns and 'Parch' in df.columns:
                df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            
            # 3. Log transform Fare
            if 'Fare' in df.columns:
                df['Fare'] = np.log(df['Fare'] + 1)
            
            # 4. Encode Sex (male=1, female=0)
            if 'Sex' in df.columns:
                df['Sex'] = df['Sex'].map({"male": 1, "female": 0})
            
            # 5. One-hot encode Embarked (drop_first=True means we drop first category C)
            if 'Embarked' in df.columns:
                df['Embarked'].fillna('S', inplace=True)
                embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True, dtype=int)
                df = pd.concat([df, embarked_dummies], axis=1)
                df = df.drop('Embarked', axis=1)
            
            # 6. One-hot encode Title (drop_first=True)
            # Title categories: Mr, Mrs, Miss, Master, Rare (5 categories, so 4 after drop_first)
            if 'Title' in df.columns:
                title_dummies = pd.get_dummies(df['Title'], prefix='Title', drop_first=True, dtype=int)
                df = pd.concat([df, title_dummies], axis=1)
                df = df.drop('Title', axis=1)
            
            # 7. Reorder and ensure correct columns for the model
            # Features MUST be in this exact order and format as model was trained:
            numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'FamilySize']
            # Sex is numeric (encoded as 1/0)
            sex_feature = ['Sex']
            # One-hot encoded features (alphabetically ordered by get_dummies)
            # Embarked_Q, Embarked_S (C is dropped)
            # Title_Master, Title_Miss, Title_Mrs, Title_Rare (Mr is dropped)
            embarked_features = ['Embarked_Q', 'Embarked_S']
            title_features = ['Title_Master', 'Title_Miss', 'Title_Mrs', 'Title_Rare']
            
            expected_cols = numeric_features + sex_feature + embarked_features + title_features
            
            # Ensure all columns exist and fill missing with 0
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # Select only the expected columns in correct order
            df = df[expected_cols]
            
            logging.info(f"Preprocessed features ({len(expected_cols)}): {df.columns.tolist()}")
            logging.info(f"Feature values: {df.values}")
            
            return df.values  # Return numpy array for model prediction
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Helper class to get prediction data"""
    def __init__(self, Pclass, Sex, Age, Fare, SibSp, Parch, Title, Embarked):
        self.Pclass = Pclass
        self.Sex = Sex
        self.Age = Age
        self.Fare = Fare
        self.SibSp = SibSp
        self.Parch = Parch
        self.Title = Title
        self.Embarked = Embarked
    
    def get_data_as_dataframe(self):
        custom_data_input_dict = {
            'Pclass': [self.Pclass],
            'Sex': [self.Sex],
            'Age': [self.Age],
            'Fare': [self.Fare],
            'SibSp': [self.SibSp],
            'Parch': [self.Parch],
            'Title': [self.Title],
            'Embarked': [self.Embarked]
        }
        
        return pd.DataFrame(custom_data_input_dict)
