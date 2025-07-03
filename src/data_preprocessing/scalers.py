import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler


class PowerTransformerScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = PowerTransformer(method="yeo-johnson", standardize=True)

    def fit(self, X, y=None):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numeric_columns])
        self.numeric_columns = numeric_columns
        return self

    def transform(self, X):
        df = X.copy()
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        return df

    def inverse_transform(self, X):
        """Inverse transform the data back to original scale"""
        df = X.copy()
        df[self.numeric_columns] = self.scaler.inverse_transform(df[self.numeric_columns])
        return df


class QuantileTransformerScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = QuantileTransformer(
            output_distribution="uniform", random_state=42
        )

    def fit(self, X, y=None):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numeric_columns])
        self.numeric_columns = numeric_columns
        return self

    def transform(self, X):
        df = X.copy()
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        return df

    def inverse_transform(self, X):
        """Inverse transform the data back to original scale"""
        df = X.copy()
        df[self.numeric_columns] = self.scaler.inverse_transform(df[self.numeric_columns])
        return df


class RobustTransformerScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numeric_columns])
        self.numeric_columns = numeric_columns
        return self

    def transform(self, X):
        df = X.copy()
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])
        return df

    def inverse_transform(self, X):
        """Inverse transform the data back to original scale"""
        df = X.copy()
        df[self.numeric_columns] = self.scaler.inverse_transform(df[self.numeric_columns])
        return df
