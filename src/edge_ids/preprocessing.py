# Shared preprocessing and feature-reduction pipeline

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

# Metadata recorded with each trained model for reproducibility
@dataclass
class PreprocessMetadata:
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_reduction_enabled: bool
    selected_feature_count: int | None
    transformed_feature_count: int

# Fit and apply one shared feature transformation to all model types
class FeaturePipeline:
    def __init__(self, feature_reduction_enabled: bool = False, k: int = 64):
        if feature_reduction_enabled and k <= 0:
            raise ValueError('Feature-reduction k must be greater than zero.')
        self.feature_reduction_enabled = feature_reduction_enabled
        self.k = k
        self.preprocessor: ColumnTransformer | None = None
        self.selector: SelectKBest | None = None
        self.metadata: PreprocessMetadata | None = None

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [c for c in X.columns if c not in numeric_columns]

        numeric_steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
        categorical_steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]

        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_columns:
            transformers.append(('numeric', Pipeline(numeric_steps), numeric_columns))
        if categorical_columns:
            transformers.append(('categorical', Pipeline(categorical_steps), categorical_columns))
        if not transformers:
            raise ValueError('No usable feature columns found after preprocessing.')

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeaturePipeline':
        self.preprocessor = self._build_preprocessor(X)
        transformed = self.preprocessor.fit_transform(X, y)
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
            transformed_feature_count=transformed_feature_count
        )
        return self 

    # Transform features using only state learned from the training split
    def transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            raise RuntimeError('FeaturePipeline has not been fitted.')

        transformed = self.preprocessor.transform(X)
        if self.selector is not None:
            transformed = self.selector.transform(transformed)
        return transformed 

    def fit_transform(self, X: pd.DataFrame, y: pd.Series): 
        self.fit(X, y)
        return self.transform(X)

    @property
    def output_feature_count(self) -> int:
        if self.metadata is None:
            raise RuntimeError('FeaturePipeline has not been fitted.')
        if self.metadata.selected_feature_count is not None:
            return self.metadata.selected_feature_count
        return self.metadata.transformed_feature_count
