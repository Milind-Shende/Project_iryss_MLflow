import pandas as pd
import os
from IryssMLflow import logger
import joblib
from IryssMLflow.entity.config_entity import ModelTrainerConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column].values
        test_y = test_data[self.config.target_column].values

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
        numeric_transformer = Pipeline(steps=[('robust_scaler', RobustScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))])

        # Numerical and Categorical Column Transformation
        logger.info("Numerical and Categorical Column Transformation")
        transformer = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                      ('cat', categorical_transformer, categorical_features)])
        logger.info("Transforming the Train and Test")
        train_x = transformer.fit_transform(train_x)
        test_x = transformer.transform(test_x)

        logger.info("define Xgboost regression")
        xgb_model = xgb.XGBRegressor(
            objective=self.config.objective,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,
            random_state=self.config.random_state
        )

        logger.info("fit train_x and train_y")
        xgb_model.fit(train_x, train_y)

        logger.info("Creating a pickle file")
        joblib.dump(xgb_model, os.path.join(self.config.root_dir, self.config.model_name))