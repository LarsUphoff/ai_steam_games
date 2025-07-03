import pandas as pd
import ast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.weekday_encoder = OneHotEncoder(sparse_output=False)
        self.developer_tier_encoder = OneHotEncoder(sparse_output=False)
        self.weekday_feature_names = None
        self.developer_tier_feature_names = None

    def fit(self, X, y=None):
        df = X.copy()
        self.weekday_encoder.fit(df[["weekday"]])
        self.weekday_feature_names = [
            f"weekday_{cat}" for cat in self.weekday_encoder.categories_[0]
        ]

        self.developer_tier_encoder.fit(df[["developer_tier"]])
        self.developer_tier_feature_names = [
            f"developer_tier_{cat}" for cat in self.developer_tier_encoder.categories_[0]
        ]
        return self

    def transform(self, X):
        df = X.copy()
        weekday_encoded = self.weekday_encoder.transform(df[["weekday"]])
        weekday_df = pd.DataFrame(
            weekday_encoded, columns=self.weekday_feature_names, index=df.index
        )

        developer_tier_encoded = self.developer_tier_encoder.transform(df[["developer_tier"]])
        developer_tier_df = pd.DataFrame(
            developer_tier_encoded, columns=self.developer_tier_feature_names, index=df.index
        )

        df = df.drop(columns=["weekday", "developer_tier"])
        return pd.concat([df, weekday_df, developer_tier_df], axis=1)


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.genres_tags_encoder = MultiLabelBinarizer()
        self.categories_encoder = MultiLabelBinarizer()
        self.supported_languages_encoder = MultiLabelBinarizer()
        self.full_audio_languages_encoder = MultiLabelBinarizer()
        self.genres_tags_feature_names = None
        self.categories_feature_names = None
        self.supported_languages_feature_names = None
        self.full_audio_languages_feature_names = None

    def fit(self, X, y=None):
        df = X.copy()

        genres_tags_lists = df["genres_tags"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.genres_tags_encoder.fit(genres_tags_lists)

        categories_lists = df["categories"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.categories_encoder.fit(categories_lists)

        supported_languages_lists = df["supported_languages"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.supported_languages_encoder.fit(supported_languages_lists)

        full_audio_languages_lists = df["full_audio_languages"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        self.full_audio_languages_encoder.fit(full_audio_languages_lists)

        self.genres_tags_feature_names = [
            f"genres_tags_{cat}" for cat in self.genres_tags_encoder.classes_
        ]
        self.categories_feature_names = [
            f"categories_{cat}" for cat in self.categories_encoder.classes_
        ]
        self.supported_languages_feature_names = [
            f"supported_languages_{cat}"
            for cat in self.supported_languages_encoder.classes_
        ]
        self.full_audio_languages_feature_names = [
            f"full_audio_languages_{cat}"
            for cat in self.full_audio_languages_encoder.classes_
        ]

        return self

    def transform(self, X):
        df = X.copy()

        genres_tags_lists = df["genres_tags"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        genres_tags_encoded = self.genres_tags_encoder.transform(genres_tags_lists)

        categories_lists = df["categories"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        categories_encoded = self.categories_encoder.transform(categories_lists)

        supported_languages_lists = df["supported_languages"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        supported_languages_encoded = self.supported_languages_encoder.transform(
            supported_languages_lists
        )

        full_audio_languages_lists = df["full_audio_languages"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        full_audio_languages_encoded = self.full_audio_languages_encoder.transform(
            full_audio_languages_lists
        )

        genres_tags_df = pd.DataFrame(
            genres_tags_encoded, columns=self.genres_tags_feature_names, index=df.index
        )
        categories_df = pd.DataFrame(
            categories_encoded, columns=self.categories_feature_names, index=df.index
        )
        supported_languages_df = pd.DataFrame(
            supported_languages_encoded,
            columns=self.supported_languages_feature_names,
            index=df.index,
        )
        full_audio_languages_df = pd.DataFrame(
            full_audio_languages_encoded,
            columns=self.full_audio_languages_feature_names,
            index=df.index,
        )

        df = df.drop(
            columns=[
                "genres_tags",
                "categories",
                "supported_languages",
                "full_audio_languages",
            ]
        )

        return pd.concat(
            [
                df,
                genres_tags_df,
                categories_df,
                supported_languages_df,
                full_audio_languages_df,
            ],
            axis=1,
        )
