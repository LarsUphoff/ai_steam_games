from __future__ import annotations
import pandas as pd
import ast
import re
from typing import Any, Dict, List, Sequence
from sklearn.base import BaseEstimator, TransformerMixin
from metadata_options import get_developer_tiers


class DeveloperTierClassifier(BaseEstimator, TransformerMixin):
    _LIST_LIKE = re.compile(r"^\s*[\[\(\{].*[\]\)\}]\s*$")

    def safe_literal_eval(self, x):
        if pd.isna(x) or x == "" or x == "[]":
            return []
        try:
            items = ast.literal_eval(x)
            return sorted([item.lower() for item in items])
        except (ValueError, SyntaxError):
            return []

    def __init__(
        self,
        tier_mapping: Dict[str, Sequence[str]] | None = None,
        tier_order: Sequence[str] = ("aaa", "aa+", "aa", "a"),
    ) -> None:
        self.tier_mapping = tier_mapping
        self.tier_order = tuple(tier_order)
        self._tier_sets: Dict[str, set[str]] | None = None

    @staticmethod
    def _normalise_name(name: Any) -> str:
        return str(name).strip().lower()

    def _prepare_tier_sets(self) -> Dict[str, set[str]]:
        mapping = self.tier_mapping or get_developer_tiers()
        return {
            tier: {self._normalise_name(dev) for dev in mapping.get(tier, [])}
            for tier in self.tier_order
        }

    def _parse_developers(self, cell: Any) -> List[str]:
        if pd.isna(cell):
            return []

        if isinstance(cell, (list, set, tuple)):
            return [self._normalise_name(d) for d in cell]

        if isinstance(cell, str) and self._LIST_LIKE.match(cell):
            try:
                parsed = ast.literal_eval(cell)
                if isinstance(parsed, (list, set, tuple)):
                    return [self._normalise_name(d) for d in parsed]
            except (ValueError, SyntaxError):
                pass

        cleaned = str(cell).translate(str.maketrans("", "", "[](){}'\""))
        return [
            self._normalise_name(d) for d in re.split(r"[;,]", cleaned) if d.strip()
        ]

    def fit(self, X: pd.DataFrame, y: Any = None):
        self._tier_sets = self._prepare_tier_sets()
        return self

    def _mark_developer_tier(
        self,
        df: pd.DataFrame,
        dev_col: str = "developers",
        tier_col: str = "developer_tier",
    ) -> pd.DataFrame:
        tier_sets = self._tier_sets or {}

        def classify(cell: Any) -> str:
            devs = self._parse_developers(cell)
            for tier in self.tier_order:
                if any(dev in tier_sets.get(tier, set()) for dev in devs):
                    return tier
            return "indie"

        df[tier_col] = df[dev_col].apply(classify)
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._tier_sets is None:
            raise RuntimeError(
                "FeatureEngineer instance is not fitted yet. "
                "Call fit() before transform()."
            )

        df = X.copy(deep=True)
        return self._mark_developer_tier(df)
