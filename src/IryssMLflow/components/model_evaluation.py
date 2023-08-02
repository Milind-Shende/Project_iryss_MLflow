import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
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
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    


    def log_into_mlflow(self):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        numeric_features = ['certificate_id', 'zipcode', 'length_of_Stay', 'discharge_year', 'drg_code',
                            'severity_of_illness_code']

        categorical_features = ['hospital_service_area', 'hospital_country', 'facility_name', 'age_group', 'gender',
                                'race', 'ethnicity', 'type_of_admission', 'patient_disposition', 'procedure_code',
                                'procedure_description', 'drg_description', 'mdc_description',
                                'severity_of_illness_description', 'risk_of_mortality',
                                'medical_surgical_description', 'payment_typology_1', 'payment_typology_2',
                                'payment_typology_3', 'emergency_department_indicator', 'diagnosis_description']

        # Numerical and Categorical Pipeline Transformation
        logger.info("Numerical and Categorical Pipeline Transformation")
        numeric_transformer = Pipeline(steps=[('Minmax_scaler', MinMaxScaler(feature_range=(-1 , 1)))])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))])

        # Numerical and Categorical Column Transformation
        logger.info("Numerical and Categorical Column Transformation")
        transformer = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                      ('cat', categorical_transformer, categorical_features)])
        logger.info("Transforming the Train and Test")
        train_x = transformer.fit_transform(train_x)
        test_x = transformer.transform(test_x)

        logger.info("Scaling train_y and test_y")
        scaler=MinMaxScaler(feature_range=(0 , 1))
        train_y=scaler.fit_transform(train_y.values.reshape(-1,1))
        test_y=scaler.fit_transform(test_y.values.reshape(-1,1))


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="xgboostModel")
            else:
                mlflow.sklearn.log_model(model, "model")

    
