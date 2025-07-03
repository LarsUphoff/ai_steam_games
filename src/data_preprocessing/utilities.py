import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureNameCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df.columns = df.columns.str.replace(r"[\[\]<>]", "", regex=True)
        return df


class FilterForIndieGames(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        indie_mask = (
            (df.get("genres_tags_indie", 0) == 1)
            | (df.get("genres_tags_crowdfunded", 0) == 1)
            | (df.get("genres_tags_kickstarter", 0) == 1)
        )

        no_higher_tier_developer = df["developer_tier_indie"] == 1
        price_ok = df.get("price", float("inf")) <= 30

        audio_cols = [c for c in df.columns if c.startswith("full_audio_languages_")]
        langs_ok = (
            df[audio_cols].sum(axis=1) < 4
            if audio_cols
            else pd.Series(True, index=df.index)
        )

        mask = indie_mask & no_higher_tier_developer & price_ok & langs_ok
        return df.loc[mask].copy()
