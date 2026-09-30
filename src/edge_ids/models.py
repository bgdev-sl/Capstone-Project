from __future__ import annotations
from typing import Any
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

SUPPORTED_MODELS = ('decision_tree', 'random_forest', 'xgboost', 'mlp')

def build_model(name: str, config: dict[str, Any], random_state: int = 42):
    name = name.lower().strip()
    model_config = config.get('models', {}).get(name, {})

    if name == 'decision_tree':
        return DecisionTreeClassifier(
            max_depth=model_config.get('max_depth', 16),
            min_samples_leaf=model_config.get('min_samples_leaf', 2),
            random_state=random_state
        )

    if name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=model_config.get('n_estimators', 150),
            max_depth=model_config.get('max_depth', 18),
            min_samples_leaf=model_config.get('min_samples_leaf', 2),
            n_jobs=model_config.get('n_jobs', -1),
            random_state=random_state
        )

    if name == 'xgboost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError('xgboost is required for XGBoost model.') from exc 

        return XGBClassifier(
            n_estimators=model_config.get('n_estimators', 200),
            max_depth=model_config.get('max_depth', 8),
            learning_rate=model_config.get('learning_rate', 0.10),
            subsample=model_config.get('subsample', 0.80),
            colsample_bytree=model_config.get('colsample_bytree', 0.80),
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=random_state
        )

    if name == 'mlp':
        hidden = tuple(model_config.get('hidden_layer_sizes', [64, 32]))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=model_config.get('max_iter', 30),
            batch_size=model_config.get('batch_size', 256),
            early_stopping=model_config.get('early_stopping', True),
            random_state=random_state
        )

    raise ValueError(f"Unsupported model '{name}'. Choose one of {SUPPORTED_MODELS}.")