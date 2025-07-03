from sklearn.pipeline import Pipeline
from .base_transformers import DataLoader, FeatureEngineer
from .developer_classifier import DeveloperTierClassifier
from .data_cleaners import DataCleaner, OutlierRemover, ListProcessor, DataCleanerFinal
from .scalers import (
    PowerTransformerScaler,
    QuantileTransformerScaler,
    RobustTransformerScaler,
)
from .encoders import CategoricalEncoder, MultiLabelEncoder
from .utilities import FeatureNameCleaner, FilterForIndieGames


base_pipeline = Pipeline(
    [
        ("data_loading", DataLoader()),
        ("feature_engineering", FeatureEngineer()),
        ("data_cleaning", DataCleaner()),
        ("list_processing", ListProcessor()),
        ("developer_tier_classification", DeveloperTierClassifier()),
    ]
)

scaling_pipeline = Pipeline(
    [
        ("scaling", PowerTransformerScaler()),
        ("categorical_encoding", CategoricalEncoder()),
        ("multilabel_encoding", MultiLabelEncoder()),
        ("feature_name_cleaning", FeatureNameCleaner()),
    ]
)

final_cleaning_pipeline = Pipeline(
    [
        ("outlier_removal", OutlierRemover()),
        ("data_final_cleaning", DataCleanerFinal()),
    ]
)

plotting_pipeline = Pipeline(
    [
        ("categorical_encoding", CategoricalEncoder()),
        ("multilabel_encoding", MultiLabelEncoder()),
        ("feature_name_cleaning", FeatureNameCleaner()),
    ]
)

indie_filter_pipeline = Pipeline(
    [
        ("indie_filter", FilterForIndieGames()),
    ]
)

power_scaler = PowerTransformerScaler()
quantile_scaler = QuantileTransformerScaler()
robust_scaler = RobustTransformerScaler()
