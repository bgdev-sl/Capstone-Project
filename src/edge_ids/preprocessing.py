"""Shared preprocessing and feature-reduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

DEFAULT_MAX_ABS_NUMERIC = 1.0e12
DEFAULT_MAX_ABS_STANDARDIZED = 100.0


@dataclass
class PreprocessMetadata:
    """Metadata recorded with each trained model for reproducibility."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_reduction_enabled: bool
    selected_feature_count: int | None
    transformed_feature_count: int
    max_abs_numeric: float = DEFAULT_MAX_ABS_NUMERIC
    max_abs_standardized: float = DEFAULT_MAX_ABS_STANDARDIZED
    numeric_nonfinite_to_missing: int = 0
    numeric_values_clipped: int = 0


class FeaturePipeline:
    """Fit and apply one shared feature transformation to all model types.

    Using the same preprocessing object for all algorithms makes model-to-model
    comparison more controlled, matching the report's requirement for a cohesive
    evaluation environment.
    """

    def __init__(
        self,
        feature_reduction_enabled: bool = False,
        k: int = 64,
        max_abs_numeric: float = DEFAULT_MAX_ABS_NUMERIC,
        max_abs_standardized: float = DEFAULT_MAX_ABS_STANDARDIZED,
    ):
        if feature_reduction_enabled and k <= 0:
            raise ValueError("Feature-reduction k must be greater than zero.")
        if not np.isfinite(max_abs_numeric) or not 0 < max_abs_numeric <= 1.0e100:
            raise ValueError("max_abs_numeric must be positive, finite, and no greater than 1e100.")
        if not np.isfinite(max_abs_standardized) or not 0 < max_abs_standardized <= 1.0e6:
            raise ValueError("max_abs_standardized must be positive, finite, and no greater than 1e6.")
        self.feature_reduction_enabled = feature_reduction_enabled
        self.k = k
        self.max_abs_numeric = float(max_abs_numeric)
        self.max_abs_standardized = float(max_abs_standardized)
        self.numeric_nonfinite_to_missing = 0
        self.numeric_values_clipped = 0
        self.preprocessor: ColumnTransformer | None = None
        self.selector: SelectKBest | None = None
        self.metadata: PreprocessMetadata | None = None

    def _sanitize_numeric_input(
        self, X: pd.DataFrame, *, record_diagnostics: bool = False
    ) -> pd.DataFrame:
        """Replace numeric +/-Inf with NaN and cap extreme finite input values.

        SimpleImputer can then handle non-finite cells as missing values. Clipping
        prevents very large but finite values from overflowing scaler statistics.
        Only dtyped numeric columns are changed; categorical data remains untouched.
        """
        # Shallow copy the frame and replace only numeric columns; this avoids an
        # unnecessary deep copy of every categorical/string column in batch mode.
        cleaned = X.copy(deep=False)
        # getattr defaults preserve inference compatibility with FeaturePipeline
        # bundles created before numeric-safety settings were added in v0.5.1.
        max_abs_numeric = getattr(self, "max_abs_numeric", DEFAULT_MAX_ABS_NUMERIC)
        numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_columns:
            values = pd.to_numeric(cleaned[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            nonfinite = ~np.isfinite(values)
            if record_diagnostics:
                self.numeric_nonfinite_to_missing = getattr(self, "numeric_nonfinite_to_missing", 0) + int(nonfinite.sum())
            values[nonfinite] = np.nan
            extreme = np.isfinite(values) & (np.abs(values) > max_abs_numeric)
            if record_diagnostics:
                self.numeric_values_clipped = getattr(self, "numeric_values_clipped", 0) + int(extreme.sum())
            np.clip(values, -max_abs_numeric, max_abs_numeric, out=values)
            cleaned[column] = values
        return cleaned

    def _clip_transformed(self, transformed):
        """Ensure transformed numeric values are finite and bounded before ML input."""
        max_abs_standardized = getattr(self, "max_abs_standardized", DEFAULT_MAX_ABS_STANDARDIZED)
        if sparse.issparse(transformed):
            output = transformed.tocsr(copy=True)
            np.nan_to_num(
                output.data,
                copy=False,
                nan=0.0,
                posinf=max_abs_standardized,
                neginf=-max_abs_standardized,
            )
            np.clip(
                output.data,
                -max_abs_standardized,
                max_abs_standardized,
                out=output.data,
            )
            output.eliminate_zeros()
            if not np.isfinite(output.data).all():
                raise ValueError(
                    "Preprocessing produced non-finite values; inspect the dataset and numeric safety limits."
                )
            return output

        output = np.asarray(transformed, dtype=np.float64).copy()
        np.nan_to_num(
            output,
            copy=False,
            nan=0.0,
            posinf=max_abs_standardized,
            neginf=-max_abs_standardized,
        )
        np.clip(
            output,
            -max_abs_standardized,
            max_abs_standardized,
            out=output,
        )
        if not np.isfinite(output).all():
            raise ValueError(
                "Preprocessing produced non-finite values; inspect the dataset and numeric safety limits."
            )
        return output

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [c for c in X.columns if c not in numeric_columns]

        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]

        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_columns:
            transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
        if categorical_columns:
            transformers.append(("categorical", Pipeline(categorical_steps), categorical_columns))

        if not transformers:
            raise ValueError("No usable feature columns were found after preprocessing.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeaturePipeline":
        safe_X = self._sanitize_numeric_input(X, record_diagnostics=True)
        self.preprocessor = self._build_preprocessor(safe_X)
        transformed = self._clip_transformed(self.preprocessor.fit_transform(safe_X, y))

        transformed_feature_count = int(transformed.shape[1])
        selected_feature_count: int | None = None

        if self.feature_reduction_enabled:
            actual_k = min(self.k, transformed_feature_count)
            self.selector = SelectKBest(score_func=f_classif, k=actual_k)
            self.selector.fit(transformed, y)
            selected_feature_count = actual_k
        else:
            self.selector = None

        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [c for c in X.columns if c not in numeric_columns]
        self.metadata = PreprocessMetadata(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            feature_reduction_enabled=self.feature_reduction_enabled,
            selected_feature_count=selected_feature_count,
            transformed_feature_count=transformed_feature_count,
            max_abs_numeric=self.max_abs_numeric,
            max_abs_standardized=self.max_abs_standardized,
            numeric_nonfinite_to_missing=self.numeric_nonfinite_to_missing,
            numeric_values_clipped=self.numeric_values_clipped,
        )
        return self

    def transform(self, X: pd.DataFrame):
        """Transform features using only state learned from the training split."""
        if self.preprocessor is None:
            raise RuntimeError("FeaturePipeline has not been fitted.")

        safe_X = self._sanitize_numeric_input(X)
        transformed = self._clip_transformed(self.preprocessor.transform(safe_X))
        if self.selector is not None:
            transformed = self._clip_transformed(self.selector.transform(transformed))
        return transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        self.fit(X, y)
        return self.transform(X)

    @property
    def output_feature_count(self) -> int:
        if self.metadata is None:
            raise RuntimeError("FeaturePipeline has not been fitted.")
        if self.metadata.selected_feature_count is not None:
            return self.metadata.selected_feature_count
        return self.metadata.transformed_feature_count
