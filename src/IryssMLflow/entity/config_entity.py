from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    objective: str 
    n_estimators: int               
    learning_rate: float              
    max_depth: int                    
    min_child_weight: int             
    gamma: float                        
    subsample: int                    
    colsample_bytree: int            
    reg_lambda: int                   
    reg_alpha: int                    
    random_state: int 
    target_column: str