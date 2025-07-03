import pandas as pd
import numpy as np
import ast
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df.loc[df["publishers"] == "[]", "publishers"] = df.loc[
            df["publishers"] == "[]", "developers"
        ]
        df.loc[df["developers"] == "[]", "developers"] = df.loc[
            df["developers"] == "[]", "publishers"
        ]

        df = df[df["developers"] != "[]"]
        df = df[df["publishers"] != "[]"]

        df = df.sort_values(by="estimated_owners", ascending=False)
        df = df.drop_duplicates(
            subset=["name", "developers", "publishers"], keep="first"
        )
        df = df.drop(columns=["estimated_owners"])

        mask = df["pct_pos_total"] == -1
        zero_reviews_mask = mask & (df["positive"] == 0) & (df["negative"] == 0)
        df.loc[zero_reviews_mask, "pct_pos_total"] = 50

        total_reviews = df["positive"] + df["negative"]
        calc_mask = mask & (total_reviews > 0)
        df["pct_pos_total"] = df["pct_pos_total"].astype(float)
        df.loc[calc_mask, "pct_pos_total"] = (
            df.loc[calc_mask, "positive"] / total_reviews[calc_mask] * 100
        ).round(2)

        return df


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        bounds = {
            "price": (0, 100),
            "dlc_count": (0, 20),
            "achievements": (0, 100),
            "average_playtime_forever": (0, 6000),
            "median_playtime_forever": (0, 6500),
            "screenshot_count": (0, 50),
            "movie_count": (0, 25),
            "description_word_count": (0, 1000),
            "estimated_owners_calculated": (0, 2500000),
        }

        masks = [
            (df[col] >= min_val) & (df[col] <= max_val)
            for col, (min_val, max_val) in bounds.items()
        ]

        combined_mask = np.ones(len(df), dtype=bool)
        for mask in masks:
            combined_mask = combined_mask & mask

        return df[combined_mask].reset_index(drop=True)


class ListProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def safe_literal_eval(self, x):
        if pd.isna(x) or x == "" or x == "[]":
            return []
        try:
            items = ast.literal_eval(x)
            return sorted([item.lower() for item in items])
        except (ValueError, SyntaxError):
            return []

    def transform(self, X):
        df = X.copy()

        list_columns = [
            "categories",
            "supported_languages",
            "full_audio_languages",
            "genres_tags",
        ]

        for col in list_columns:
            df[col] = df[col].apply(self.safe_literal_eval)

        df["full_audio_languages"] = df["full_audio_languages"].apply(
            lambda x: x if x else ["No full audio support"]
        )

        return df


class DataCleanerFinal(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        columns_to_drop_final = [
            "appid",
            "name",
            "developers",
            "publishers",
            "positive",
            "negative",
            "average_playtime_2weeks",
            "median_playtime_2weeks",
            "pct_pos_total",
            "num_reviews_total",
        ]

        df = df.drop(columns=columns_to_drop_final)
        return df.dropna()
