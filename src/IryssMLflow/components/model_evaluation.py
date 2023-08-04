import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.xgboost
import numpy as np
import joblib
from IryssMLflow.entity.config_entity import ModelEvaluationConfig
from IryssMLflow.utils.common import save_json
from pathlib import Path
from IryssMLflow import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler,MinMaxScaler,StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        mse=mean_squared_error(actual,pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return mse,rmse, mae, r2
    


    def log_into_mlflow(self):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        transformer = joblib.load(self.config.transformer_path)
        target = joblib.load(self.config.target_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

    
        logger.info("Transforming the Train and Test")
        train_x = transformer.fit_transform(train_x)
        test_x = transformer.transform(test_x)

        logger.info("Scaling train_y and test_y")
        
        train_y=target.fit_transform(train_y.values.reshape(-1,1))
        test_y=target.fit_transform(test_y.values.reshape(-1,1))


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities_train = model.predict(train_x)

            (mse_train,rmse_train, mae_train, r2_train) = self.eval_metrics(train_y, predicted_qualities_train)
            
            # Saving metrics as local
            scores_train = {"mse_train":mse_train,"rmse_train": rmse_train, "mae_train": mae_train, "r2_train": r2_train}
            save_json(path=Path(self.config.metric_file_name_train), data=scores_train)



            predicted_qualities_test = model.predict(test_x)

            (mse_test,rmse_test, mae_test, r2_test) = self.eval_metrics(test_y, predicted_qualities_test)
            
            # Saving metrics as local
            scores_test = {"mse_test":mse_test,"rmse_test": rmse_test, "mae_test": mae_test, "r2_test": r2_test}
            save_json(path=Path(self.config.metric_file_name_test), data=scores_test)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("mse_train", mse_train)
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("r2_train", r2_train)


            mlflow.log_metric("mse_test", mse_test)
            mlflow.log_metric("rmse_test", rmse_test)
            mlflow.log_metric("mae_test", mae_test)
            mlflow.log_metric("r2_test", r2_test)
            


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.xgboost.log_model(model, "model", registered_model_name="xgboostModel")
            else:
                mlflow.xgboost.log_model(model, "model")

    
