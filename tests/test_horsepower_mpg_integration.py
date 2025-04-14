"""Integration test using horsepower to predict MPG, comparing FASTER with baseline models for regression."""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
from typing import Dict, List, Tuple
from unittest.mock import patch
import os
import openai
import logging
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from faster.pipeline import Pipeline, PipelineConfig
from faster.feature_selection import SelectionCriteria
from faster.domain_knowledge import DomainKnowledgeExtractor
from tests.conftest import colored_metric_output

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MOCK_LLM_RESPONSE = [
    {
        "feature_name": "horsepower",
        "importance": 0.9,
        "relationships": [],
        "suggested_transformations": ["log", "zscore", "polynomial", "square_root", "reciprocal"],
        "rationale": "Horsepower has an inverse relationship with MPG. More powerful engines consume more fuel. Various transformations like log and polynomial can help capture the non-linear relationship between horsepower and fuel efficiency.",
    }
]


def load_and_preprocess_horsepower_mpg() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Auto MPG dataset, keeping only horsepower feature and mpg target."""
    try:
        auto_mpg = fetch_openml(name="auto-mpg", version=1, as_frame=True)

        df = auto_mpg.data
        df["mpg"] = auto_mpg.target

    except Exception as e:
        logger.warning(f"Error fetching from OpenML: {str(e)}. Falling back to local loading...")

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        column_names = [
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model_year",
            "origin",
            "car_name",
        ]
        df = pd.read_csv(url, delim_whitespace=True, header=None, names=column_names, na_values="?")

    df = df[["horsepower", "mpg"]]

    imputer = SimpleImputer(strategy="median")
    df["horsepower"] = imputer.fit_transform(df[["horsepower"]])

    y = df["mpg"]
    X = df[["horsepower"]]

    scaler = StandardScaler()
    X["horsepower"] = scaler.fit_transform(X[["horsepower"]])

    return X, y


def evaluate_model_with_split(
    model, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate regression model using train-test split to check for overfitting.

    Args:
        model: The scikit-learn model to evaluate
        X: Feature dataframe
        y: Target series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Dict containing train and test metrics
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    r2_diff = train_r2 - test_r2
    mae_diff = test_mae - train_mae
    rmse_diff = test_rmse - train_rmse

    return {
        "test_r2": test_r2,
        "test_mae": -test_mae,  # Negate for consistency with cross_validate
        "test_rmse": -test_rmse,  # Negate for consistency with cross_validate
        "train_r2": train_r2,
        "train_mae": -train_mae,  # Negate for consistency with cross_validate
        "train_rmse": -train_rmse,  # Negate for consistency with cross_validate
        "r2_diff": r2_diff,
        "mae_diff": mae_diff,
        "rmse_diff": rmse_diff,
    }


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate regression model using cross-validation."""

    regressor = Ridge(alpha=1.0, random_state=42)

    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(regressor, X, y, cv=5, scoring=scoring, return_train_score=True)

    return {
        "test_r2": cv_results["test_r2"].mean(),
        "test_mae": -cv_results["test_mae"].mean(),
        "test_rmse": -cv_results["test_rmse"].mean(),
        "train_r2": cv_results["train_r2"].mean(),
        "train_mae": -cv_results["train_mae"].mean(),
        "train_rmse": -cv_results["train_rmse"].mean(),
        "r2_diff": cv_results["train_r2"].mean() - cv_results["test_r2"].mean(),
        "mae_diff": -cv_results["test_mae"].mean() - (-cv_results["train_mae"].mean()),
        "rmse_diff": -cv_results["test_rmse"].mean() - (-cv_results["train_rmse"].mean()),
    }


def verify_openrouter_api_key() -> str:
    """
    Verify and retrieve the OpenRouter API key.

    Returns:
        str: Validated API key or raises a pytest.skip exception
    """

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        pytest.skip("Skipping non-mock test: OPENROUTER_API_KEY environment variable not set")

    if not api_key.startswith("sk-"):
        logger.warning("Warning: OpenRouter API key does not start with 'sk-'. Key may be invalid.")

    placeholder_terms = ["dummy", "test", "placeholder", "your_key", "example"]
    if any(term in api_key.lower() for term in placeholder_terms):
        pytest.skip(f"Skipping non-mock test: OPENROUTER_API_KEY appears to be a placeholder value")

    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    logger.info(f"Using OpenRouter API key: {masked_key} (length: {len(api_key)})")

    return api_key


def configure_openai_client(api_key: str) -> openai.OpenAI:
    """
    Configure and return an OpenAI client configured for OpenRouter.

    Args:
        api_key: The OpenRouter API key

    Returns:
        openai.OpenAI: Configured client
    """

    logger.info(f"Configuring OpenAI client with base URL: https://openrouter.ai/api/v1")

    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def check_overfitting(scores: Dict[str, float], threshold: float = 0.2) -> bool:
    """
    Check if model is overfitting based on performance metrics.

    Args:
        scores: Dictionary of performance metrics
        threshold: Threshold for R² difference that indicates overfitting

    Returns:
        bool: True if model is overfitting, False otherwise
    """
    r2_diff = scores.get("r2_diff", 0)

    return r2_diff > threshold


@pytest.mark.parametrize("use_mock", [True, False])
def test_horsepower_mpg_integration(use_mock):
    """Integration test comparing baseline and FASTER-enhanced models using only horsepower to predict MPG."""

    if not use_mock:
        try:
            api_key = verify_openrouter_api_key()

            client = configure_openai_client(api_key)

            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-chat:free",
                    messages=[{"role": "user", "content": "Hello, this is a test"}],
                    temperature=0.0,
                    max_tokens=10,
                )
                logger.info("OpenRouter API test successful")
            except Exception as e:
                logger.error(f"OpenRouter API test failed: {str(e)}")
                pytest.skip(f"Skipping non-mock test: OpenRouter API request failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error setting up OpenRouter credentials: {str(e)}")
            pytest.skip(f"Skipping non-mock test: {str(e)}")

    X, y = load_and_preprocess_horsepower_mpg()
    data = X.copy()
    data["mpg"] = y

    linear_model = Ridge(alpha=1.0)
    linear_scores = evaluate_model_with_split(linear_model, X, y)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_scores = evaluate_model_with_split(rf_model, X, y)

    config = PipelineConfig(
        model_name="deepseek/deepseek-chat:free",
        temperature=0.0,
        max_interaction_degree=1,  # Limit interactions since we only have one feature
        text_feature_method="tfidf",
        alpha=0.05,
        correction_method="fdr_bh",
        selection_criteria=SelectionCriteria(),
        output_dir=None,
        save_intermediate=False,
        categorical_encoding="dummy",
        keep_all_features=True,
    )

    if use_mock:
        with patch(
            "faster.domain_knowledge.DomainKnowledgeExtractor._query_llm_with_retry"
        ) as mock_query:
            mock_query.return_value = MOCK_LLM_RESPONSE

            pipeline = Pipeline(config)
            result = pipeline.run(
                data=data,
                target_column="mpg",
                problem_description="""
                Regression problem predicting miles per gallon (MPG) of automobiles based only on horsepower.
                
                Domain context:
                - Higher horsepower engines typically consume more fuel, resulting in lower MPG
                - The relationship between horsepower and MPG is generally non-linear
                - There may be diminishing returns where increases in horsepower above certain thresholds have less impact on MPG
                """,
                is_classification=False,
                keep_all_features=True,
            )
    else:
        domain_extractor = DomainKnowledgeExtractor(
            model_name=config.model_name, temperature=config.temperature, api_key=api_key
        )

        pipeline = Pipeline(config)
        pipeline.domain_extractor = domain_extractor
        result = pipeline.run(
            data=data,
            target_column="mpg",
            problem_description="""
            Regression problem predicting miles per gallon (MPG) of automobiles based only on horsepower.
            
            Domain context:
            - Higher horsepower engines typically consume more fuel, resulting in lower MPG
            - The relationship between horsepower and MPG is generally non-linear
            - There may be diminishing returns where increases in horsepower above certain thresholds have less impact on MPG
            """,
            is_classification=False,
            keep_all_features=True,
        )

    faster_features = result.transformed_data

    if "mpg" in faster_features.columns:
        faster_features = faster_features.drop("mpg", axis=1)

    print("\nOriginal X Shape:", X.shape)
    print("FASTER Features Shape:", faster_features.shape)
    print("\nOriginal X Columns:")
    print(sorted(X.columns.tolist()))
    print("\nFASTER X Columns:")
    print(sorted(faster_features.columns.tolist()))

    faster_scores = evaluate_model_with_split(Ridge(alpha=1.0), faster_features, y)

    linear_overfitting = check_overfitting(linear_scores)
    rf_overfitting = check_overfitting(rf_scores)
    faster_overfitting = check_overfitting(faster_scores)

    print(f"\nModel Performance Comparison ({'Mock' if use_mock else 'Real'} LLM):")

    print("\nLinear Model (Ridge):")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in linear_scores.items() if k.startswith("train")]:
        print(f"    {metric.replace('train_', '')}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in linear_scores.items() if k.startswith("test")]:
        print(f"    {metric.replace('test_', '')}: {score:.4f}")
    print(f"  Overfitting: {'Yes' if linear_overfitting else 'No'}")
    print(f"  R² Difference: {linear_scores['r2_diff']:.4f}")

    print("\nRandom Forest Model:")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in rf_scores.items() if k.startswith("train")]:
        print(f"    {metric.replace('train_', '')}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in rf_scores.items() if k.startswith("test")]:
        print(f"    {metric.replace('test_', '')}: {score:.4f}")
    print(f"  Overfitting: {'Yes' if rf_overfitting else 'No'}")
    print(f"  R² Difference: {rf_scores['r2_diff']:.4f}")

    print("\nFASTER Model (compared to Linear):")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in faster_scores.items() if k.startswith("train")]:
        metric_name = metric.replace("train_", "")
        if metric in linear_scores:
            baseline_score = linear_scores[metric]
            print(f"    {colored_metric_output(metric_name, score, baseline_score)}")
        else:
            print(f"    {metric_name}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in faster_scores.items() if k.startswith("test")]:
        metric_name = metric.replace("test_", "")
        if metric in linear_scores:
            baseline_score = linear_scores[metric]
            print(f"    {colored_metric_output(metric_name, score, baseline_score)}")
        else:
            print(f"    {metric_name}: {score:.4f}")

    print("\nFASTER Model (compared to Random Forest):")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in faster_scores.items() if k.startswith("train")]:
        metric_name = metric.replace("train_", "")
        if metric in rf_scores:
            baseline_score = rf_scores[metric]
            print(f"    {colored_metric_output(metric_name, score, baseline_score)}")
        else:
            print(f"    {metric_name}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in faster_scores.items() if k.startswith("test")]:
        metric_name = metric.replace("test_", "")
        if metric in rf_scores:
            baseline_score = rf_scores[metric]
            print(f"    {colored_metric_output(metric_name, score, baseline_score)}")
        else:
            print(f"    {metric_name}: {score:.4f}")

    print(f"  Overfitting: {'Yes' if faster_overfitting else 'No'}")
    print(f"  R² Difference: {faster_scores['r2_diff']:.4f}")

    print("\nGenerated Features from FASTER:")
    print(result.selected_features)

    assert result.selected_features is not None
    assert len(result.selected_features) > 0

    assert faster_scores["r2_diff"] <= rf_scores["r2_diff"] + 0.1, (
        "FASTER is overfitting more than expected"
    )

    all_scores = {"linear": linear_scores, "random_forest": rf_scores, "faster": faster_scores}

    print("\nAll performance metrics:", all_scores)

    return result, all_scores
