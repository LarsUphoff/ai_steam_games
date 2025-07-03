from __future__ import annotations
import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin


class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, filepath="../data/raw/games.csv"):
        self.filepath = filepath

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        df = pd.read_csv(self.filepath)

        columns_to_drop = [
            "required_age",
            "detailed_description",
            "short_description",
            "reviews",
            "header_image",
            "website",
            "support_url",
            "support_email",
            "metacritic_url",
            "recommendations",
            "notes",
            "packages",
            "user_score",
            "score_rank",
            "num_reviews_recent",
            "pct_pos_recent",
        ]

        return df.drop(columns=columns_to_drop, errors="ignore")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def extract_genres_tags(self, row):
        genres = ast.literal_eval(row["genres"])
        tags = ast.literal_eval(row["tags"])

        tag_names = list(tags.keys()) if isinstance(tags, dict) else []
        combined = list(set(genres + tag_names))

        return "[" + ", ".join([f'"{item}"' for item in combined]) + "]"

    def transform(self, X):
        df = X.copy()

        # Media counts
        df["screenshot_count"] = (
            df["screenshots"]
            .fillna("")
            .apply(lambda x: len(x.split(",")) if x and x.strip() else 0)
        )

        df["movie_count"] = (
            df["movies"]
            .fillna("")
            .apply(lambda x: len(x.split(",")) if x and x.strip() else 0)
        )

        # Combine genres and tags
        df["genres_tags"] = df.apply(self.extract_genres_tags, axis=1)

        # Calculate actual price with discount
        discount_factor = (100 - df["discount"].fillna(0)) / 100
        df["price"] = (df["price"].fillna(0) / discount_factor).round(2)

        # Total reviews and estimated owners
        df["num_reviews_total"] = (
            df["positive"].fillna(0) + df["negative"].fillna(0)
        ).astype(int)

        df["description_word_count"] = (
            df["about_the_game"]
            .fillna("")
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.split()
            .str.len()
            .fillna(0)
        )

        df["platform_count"] = df[["windows", "mac", "linux"]].sum(axis=1)

        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["weekday"] = df["release_date"].dt.day_name()

        current_date = pd.Timestamp.now()
        years_since_release = (current_date - df["release_date"]).dt.days / 365.25
        years_since_release = years_since_release.clip(lower=1)

        df["estimated_owners_calculated"] = (
            pd.concat(
                [
                    df["estimated_owners"].str.extract(r"(\d+)").astype(int),
                    df["num_reviews_total"].astype(int) * 40,
                    df["peak_ccu"].astype(int),
                ],
                axis=1,
            )
            .max(axis=1)
            .div(years_since_release)
            .astype(int)
        )

        columns_to_drop = [
            "screenshots",
            "movies",
            "genres",
            "tags",
            "discount",
            "about_the_game",
            "peak_ccu",
        ]

        return df.drop(columns=columns_to_drop)
