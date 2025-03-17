"""Evaluation utilities for FASTER Streamlit app."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline


def evaluate_model(X: pd.DataFrame, y: pd.Series, is_classification: bool) -> Dict[str, float]:
    """
    Evaluate a model using cross-validation.
    
    Args:
        X: Feature dataframe
        y: Target series
        is_classification: Whether it's a classification problem
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Encode categorical target variables if needed
    if is_classification and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipeline
    transformers = []
    
    if numerical_features:
        transformers.append(('num', StandardScaler(), numerical_features))
    
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
    
    # Only create preprocessor if we have transformers
    if transformers:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Pass through other columns rather than dropping them
        )
    else:
        preprocessor = None
    
    if is_classification:
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_key = 'classifier'
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc_ovr_weighted' if len(np.unique(y)) > 2 else 'roc_auc'
        }
    else:
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_key = 'regressor'
        scoring = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error'
        }
    
    # Create full pipeline with preprocessing and model
    if preprocessor:
        model = SklearnPipeline([
            ('preprocessor', preprocessor),
            (model_key, base_model)
        ])
    else:
        model = base_model
    
    # Perform cross-validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            model, X, y,
            cv=5,
            scoring=scoring,
            return_train_score=True
        )
    
    # Calculate mean scores
    result = {}
    
    # Process test scores
    for metric, scores in [(k.replace('test_', ''), v) for k, v in cv_results.items() if k.startswith('test_')]:
        # Negate error metrics to get positive values
        if metric in ['mae', 'rmse']:
            result[f'test_{metric}'] = -scores.mean()
        else:
            result[f'test_{metric}'] = scores.mean()
    
    # Process train scores
    for metric, scores in [(k.replace('train_', ''), v) for k, v in cv_results.items() if k.startswith('train_')]:
        # Negate error metrics to get positive values
        if metric in ['mae', 'rmse']:
            result[f'train_{metric}'] = -scores.mean()
        else:
            result[f'train_{metric}'] = scores.mean()
    
    return result 