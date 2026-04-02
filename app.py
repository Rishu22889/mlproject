from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.ml_project.components.data_transformation import DataTransformation, DataTransformationConfig

import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
    