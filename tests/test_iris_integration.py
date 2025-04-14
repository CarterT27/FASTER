"""Integration test using the Iris dataset to compare FASTER with baseline models for regression."""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
from typing import Dict, List, Tuple
from unittest.mock import patch
import os
from sklearn import datasets
import openai
import logging
from sklearn.datasets import load_iris

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
        "feature_name": "petal_length",
        "importance": 0.9,
        "relationships": ["petal_width", "species"],
        "suggested_transformations": ["log", "zscore", "polynomial"],
        "rationale": "Petal length is highly correlated with sepal length across iris species",
    },
    {
        "feature_name": "petal_width",
        "importance": 0.8,
        "relationships": ["petal_length", "species"],
        "suggested_transformations": ["log", "zscore", "polynomial"],
        "rationale": "Petal width correlates with overall flower size including sepal length",
    },
    {
        "feature_name": "species",
        "importance": 0.7,
        "relationships": ["petal_length", "petal_width"],
        "suggested_transformations": ["one_hot", "label"],
        "rationale": "Different iris species have characteristic sepal lengths",
    },
    {
        "feature_name": "sepal_width",
        "importance": 0.5,
        "relationships": ["sepal_length"],
        "suggested_transformations": ["zscore", "polynomial"],
        "rationale": "Sepal width has a moderate correlation with sepal length",
    },
]


def load_and_preprocess_iris() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Iris dataset for regression (predicting sepal length)."""

    iris = datasets.load_iris()

    feature_names = iris.feature_names
    df = pd.DataFrame(iris.data, columns=feature_names)
    df["species"] = iris.target

    species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df["species"] = df["species"].map(species_mapping)

    column_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
    df = df.rename(columns=column_mapping)

    X = df[["sepal_width", "petal_length", "petal_width", "species"]].copy()
    y = df["sepal_length"]

    X = pd.get_dummies(X, columns=["species"], drop_first=False)

    numeric_features = ["sepal_width", "petal_length", "petal_width"]
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    return X, y


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate regression model using cross-validation."""
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)

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


@pytest.mark.parametrize("use_mock", [True, False])
def test_iris_integration(use_mock):
    """Integration test comparing baseline and FASTER-enhanced models on Iris dataset for regression."""

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

    X, y = load_and_preprocess_iris()
    data = X.copy()
    data["sepal_length"] = y

    baseline_scores = evaluate_model(X, y)

    config = PipelineConfig(
        model_name="deepseek/deepseek-chat:free",
        temperature=0.0,
        max_interaction_degree=2,
        text_feature_method="tfidf",
        alpha=0.05,
        correction_method="fdr_bh",
        selection_criteria=SelectionCriteria(),
        output_dir=None,
        save_intermediate=False,
        keep_all_features=True,  # Keep all features without dropping any
    )

    if use_mock:
        with patch(
            "faster.domain_knowledge.DomainKnowledgeExtractor._query_llm_with_retry"
        ) as mock_query:
            mock_query.return_value = MOCK_LLM_RESPONSE

            pipeline_no_domain = Pipeline(config)
            result_no_domain = pipeline_no_domain.run(
                data=data,
                target_column="sepal_length",
                problem_description="Regression problem predicting sepal length",
                is_classification=False,
                keep_all_features=True,  # Keep all features without dropping any
            )

            pipeline_with_domain = Pipeline(config)
            result_with_domain = pipeline_with_domain.run(
                data=data,
                target_column="sepal_length",
                problem_description="""
                Regression problem predicting sepal length of iris flowers.
                Features:
                - sepal_width: Width of the sepal in cm
                - petal_length: Length of the petal in cm
                - petal_width: Width of the petal in cm
                - species: Type of iris (setosa, versicolor, virginica)
                
                Domain context:
                - Different iris species have characteristic sepal and petal dimensions
                - Petal dimensions tend to be correlated with sepal dimensions
                - Setosa species has the smallest petals but relatively wide sepals
                - Virginica species has the largest petals and sepals
                - There are allometric relationships between different flower parts
                """,
                is_classification=False,
                keep_all_features=True,  # Keep all features without dropping any
            )
    else:
        domain_extractor = DomainKnowledgeExtractor(
            model_name=config.model_name, temperature=config.temperature, api_key=api_key
        )

        pipeline_no_domain = Pipeline(config)
        pipeline_no_domain.domain_extractor = domain_extractor
        result_no_domain = pipeline_no_domain.run(
            data=data,
            target_column="sepal_length",
            problem_description="Regression problem predicting sepal length",
            is_classification=False,
            keep_all_features=True,  # Keep all features without dropping any
        )

        pipeline_with_domain = Pipeline(config)
        pipeline_with_domain.domain_extractor = domain_extractor
        result_with_domain = pipeline_with_domain.run(
            data=data,
            target_column="sepal_length",
            problem_description="""
            Regression problem predicting sepal length of iris flowers.
            Features:
            - sepal_width: Width of the sepal in cm
            - petal_length: Length of the petal in cm
            - petal_width: Width of the petal in cm
            - species: Type of iris (setosa, versicolor, virginica)
            
            Domain context:
            - Different iris species have characteristic sepal and petal dimensions
            - Petal dimensions tend to be correlated with sepal dimensions
            - Setosa species has the smallest petals but relatively wide sepals
            - Virginica species has the largest petals and sepals
            - There are allometric relationships between different flower parts
            """,
            is_classification=False,
            keep_all_features=True,  # Keep all features without dropping any
        )

    print("\nAvailable attributes in result_no_domain:", dir(result_no_domain))

    no_domain_features = result_no_domain.transformed_data
    with_domain_features = result_with_domain.transformed_data

    if "sepal_length" in no_domain_features.columns:
        no_domain_features = no_domain_features.drop("sepal_length", axis=1)
    if "sepal_length" in with_domain_features.columns:
        with_domain_features = with_domain_features.drop("sepal_length", axis=1)

    print("\nNo Domain Features Shape:", no_domain_features.shape)
    print("With Domain Features Shape:", with_domain_features.shape)
    print("\nBaseline X Columns:")
    print(sorted(X.columns.tolist()))
    print("\nFASTER (No Domain Knowledge) X Columns:")
    print(sorted(no_domain_features.columns.tolist()))
    print("\nFASTER (With Domain Knowledge) X Columns:")
    print(sorted(with_domain_features.columns.tolist()))

    faster_no_domain_scores = evaluate_model(no_domain_features, y)

    faster_with_domain_scores = evaluate_model(with_domain_features, y)

    print(f"\nModel Performance Comparison ({'Mock' if use_mock else 'Real'} LLM):")

    print("\nBaseline Model:")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in baseline_scores.items() if k.startswith("train")]:
        print(f"    {metric.replace('train_', '')}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in baseline_scores.items() if k.startswith("test")]:
        print(f"    {metric.replace('test_', '')}: {score:.4f}")

    print("\nFASTER (No Domain Knowledge):")
    print("  Train Metrics:")
    for metric, score in [
        (k, v) for k, v in faster_no_domain_scores.items() if k.startswith("train")
    ]:
        metric_name = metric.replace("train_", "")
        baseline_score = baseline_scores[metric]
        print(f"    {colored_metric_output(metric_name, score, baseline_score)}")
    print("  Test Metrics:")
    for metric, score in [
        (k, v) for k, v in faster_no_domain_scores.items() if k.startswith("test")
    ]:
        metric_name = metric.replace("test_", "")
        baseline_score = baseline_scores[metric]
        print(f"    {colored_metric_output(metric_name, score, baseline_score)}")

    print("\nFASTER (With Domain Knowledge):")
    print("  Train Metrics:")
    for metric, score in [
        (k, v) for k, v in faster_with_domain_scores.items() if k.startswith("train")
    ]:
        metric_name = metric.replace("train_", "")
        baseline_score = baseline_scores[metric]
        print(f"    {colored_metric_output(metric_name, score, baseline_score)}")
    print("  Test Metrics:")
    for metric, score in [
        (k, v) for k, v in faster_with_domain_scores.items() if k.startswith("test")
    ]:
        metric_name = metric.replace("test_", "")
        baseline_score = baseline_scores[metric]
        print(f"    {colored_metric_output(metric_name, score, baseline_score)}")

    print("\nGenerated Features:")
    print("\nNo Domain Knowledge:")
    print(result_no_domain.selected_features)
    print("\nWith Domain Knowledge:")
    print(result_with_domain.selected_features)

    assert result_no_domain.selected_features is not None
    assert result_with_domain.selected_features is not None
    assert len(result_no_domain.selected_features) > 0
    assert len(result_with_domain.selected_features) > 0

    all_scores = {
        "baseline": baseline_scores,
        "faster_no_domain": faster_no_domain_scores,
        "faster_with_domain": faster_with_domain_scores,
    }

    print("\nAll performance metrics:", all_scores)
