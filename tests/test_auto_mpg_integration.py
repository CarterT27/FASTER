"""Integration test using the Auto MPG dataset to compare FASTER with baseline models for regression."""

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
import openai
import logging
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer

from faster.pipeline import Pipeline, PipelineConfig
from faster.feature_selection import SelectionCriteria
from faster.domain_knowledge import DomainKnowledgeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock LLM response for Auto MPG dataset
MOCK_LLM_RESPONSE = [
    {
        "feature_name": "horsepower",
        "importance": 0.9,
        "relationships": ["weight", "cylinders", "displacement"],
        "suggested_transformations": ["log", "zscore", "polynomial", "square_root"],
        "rationale": "Horsepower has an inverse relationship with MPG. More powerful engines consume more fuel."
    },
    {
        "feature_name": "weight",
        "importance": 0.85,
        "relationships": ["horsepower", "displacement"],
        "suggested_transformations": ["log", "zscore", "polynomial", "reciprocal"],
        "rationale": "Heavier cars require more energy to move, directly impacting fuel efficiency."
    },
    {
        "feature_name": "displacement",
        "importance": 0.8,
        "relationships": ["cylinders", "horsepower"],
        "suggested_transformations": ["log", "zscore", "polynomial", "square_root"],
        "rationale": "Engine displacement affects fuel consumption; larger engines typically consume more fuel."
    },
    {
        "feature_name": "cylinders",
        "importance": 0.75,
        "relationships": ["displacement", "horsepower"],
        "suggested_transformations": ["one_hot", "zscore", "polynomial"],
        "rationale": "Number of cylinders correlates with engine size and fuel consumption."
    },
    {
        "feature_name": "model_year",
        "importance": 0.7,
        "relationships": ["weight", "horsepower"],
        "suggested_transformations": ["zscore", "binning", "time_since"],
        "rationale": "Newer models tend to have improved fuel efficiency due to technological advancements."
    },
    {
        "feature_name": "origin",
        "importance": 0.6,
        "relationships": ["model_year", "weight"],
        "suggested_transformations": ["one_hot", "label"],
        "rationale": "Cars from different regions had different design philosophies affecting fuel efficiency."
    },
    {
        "feature_name": "acceleration",
        "importance": 0.5,
        "relationships": ["horsepower", "weight"],
        "suggested_transformations": ["zscore", "polynomial", "reciprocal"],
        "rationale": "Acceleration capability relates to power-to-weight ratio, indirectly impacting fuel efficiency."
    }
]

def load_and_preprocess_auto_mpg() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Auto MPG dataset for regression (predicting MPG)."""
    try:
        # Try to fetch the dataset using fetch_openml
        auto_mpg = fetch_openml(name="auto-mpg", version=1, as_frame=True)
        
        df = auto_mpg.data
        df['mpg'] = auto_mpg.target
        
    except Exception as e:
        logger.warning(f"Error fetching from OpenML: {str(e)}. Falling back to local loading...")
        
        # Fallback to loading from UCI repository using pandas
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                         'weight', 'acceleration', 'model_year', 'origin', 'car_name']
        df = pd.read_csv(url, delim_whitespace=True, header=None, 
                         names=column_names, na_values='?')
    
    # Remove car_name column as it's just an identifier
    if 'car_name' in df.columns:
        df = df.drop('car_name', axis=1)
    
    # Handle missing values
    numeric_columns = ['horsepower']  # Only horsepower has missing values typically
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Convert 'origin' to a categorical feature with meaningful labels
    origin_mapping = {1: 'american', 2: 'european', 3: 'asian'}
    df['origin'] = df['origin'].map(origin_mapping)
    
    # For regression, we'll predict mpg based on other features
    y = df['mpg']
    X = df.drop('mpg', axis=1)
    
    # Encode categorical features
    X = pd.get_dummies(X, columns=['origin'], drop_first=False)
    
    # Scale numeric features
    numeric_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    return X, y

def evaluate_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate regression model using cross-validation."""
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Define scoring metrics for regression
    scoring = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error'
    }
    
    # Perform cross-validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            regressor, X, y,
            cv=5,
            scoring=scoring,
            return_train_score=True  # Changed to True to get train scores
        )
    
    # Calculate mean scores (negate the error metrics to get positive values)
    return {
        # Test scores
        'test_r2': cv_results['test_r2'].mean(),
        'test_mae': -cv_results['test_mae'].mean(),
        'test_rmse': -cv_results['test_rmse'].mean(),
        # Train scores
        'train_r2': cv_results['train_r2'].mean(),
        'train_mae': -cv_results['train_mae'].mean(),
        'train_rmse': -cv_results['train_rmse'].mean()
    }

def verify_openrouter_api_key() -> str:
    """
    Verify and retrieve the OpenRouter API key.
    
    Returns:
        str: Validated API key or raises a pytest.skip exception
    """
    # Check if API key exists in environment
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    
    if not api_key:
        pytest.skip("Skipping non-mock test: OPENROUTER_API_KEY environment variable not set")
    
    # Validate API key format
    if not api_key.startswith("sk-"):
        logger.warning("Warning: OpenRouter API key does not start with 'sk-'. Key may be invalid.")
    
    # Check for obvious placeholder values
    placeholder_terms = ["dummy", "test", "placeholder", "your_key", "example"]
    if any(term in api_key.lower() for term in placeholder_terms):
        pytest.skip(f"Skipping non-mock test: OPENROUTER_API_KEY appears to be a placeholder value")
    
    # Log (masked) key information for debugging
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
def test_auto_mpg_integration(use_mock):
    """Integration test comparing baseline and FASTER-enhanced models on Auto MPG dataset for regression."""
    # Set up OpenRouter credentials for non-mock tests
    if not use_mock:
        try:
            # Verify and get API key
            api_key = verify_openrouter_api_key()
            
            # Configure OpenAI client for OpenRouter
            client = configure_openai_client(api_key)
            
            # Test the client with a simple query to verify credentials work
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-chat:free",
                    messages=[{"role": "user", "content": "Hello, this is a test"}],
                    temperature=0.0,
                    max_tokens=10
                )
                logger.info("OpenRouter API test successful")
            except Exception as e:
                logger.error(f"OpenRouter API test failed: {str(e)}")
                pytest.skip(f"Skipping non-mock test: OpenRouter API request failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error setting up OpenRouter credentials: {str(e)}")
            pytest.skip(f"Skipping non-mock test: {str(e)}")

    # Load and preprocess data
    X, y = load_and_preprocess_auto_mpg()
    data = X.copy()
    data['mpg'] = y
    
    # 1. Evaluate baseline model
    baseline_scores = evaluate_model(X, y)
    
    # Configure FASTER pipeline with OpenRouter
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
        categorical_encoding="dummy",
    )

    if use_mock:
        # Mock the LLM calls
        with patch('faster.domain_knowledge.DomainKnowledgeExtractor._query_llm_with_retry') as mock_query:
            # Set up mock responses
            mock_query.return_value = MOCK_LLM_RESPONSE
            
            # Run tests with mocked LLM
            pipeline_no_domain = Pipeline(config)
            result_no_domain = pipeline_no_domain.run(
                data=data,
                target_column='mpg',
                problem_description="Regression problem predicting miles per gallon (MPG) of automobiles",
                is_classification=False
            )
            
            pipeline_with_domain = Pipeline(config)
            result_with_domain = pipeline_with_domain.run(
                data=data,
                target_column='mpg',
                problem_description="""
                Regression problem predicting miles per gallon (MPG) of automobiles.
                Features:
                - cylinders: Number of cylinders in the engine (integer)
                - displacement: Engine displacement in cubic inches (continuous)
                - horsepower: Engine horsepower (continuous)
                - weight: Vehicle weight in pounds (continuous)
                - acceleration: Time to accelerate from 0 to 60 mph in seconds (continuous)
                - model_year: Vehicle model year (integer, 70-82 representing 1970-1982)
                - origin: Origin of car (categorical: american, european, asian)
                
                Domain context:
                - Fuel efficiency (MPG) generally decreases with vehicle weight
                - Higher horsepower engines typically consume more fuel
                - Larger engine displacement generally correlates with lower MPG
                - Cars with more cylinders typically have lower fuel efficiency
                - Technological improvements over model years generally increased fuel efficiency
                - There are tradeoffs between performance (acceleration) and fuel efficiency
                - Different regions (origins) had different emission standards and design philosophies
                - The dataset contains cars from the 1970s and early 1980s during fuel crises
                """,
                is_classification=False
            )
    else:
        # Create domain knowledge extractor with OpenRouter configuration
        domain_extractor = DomainKnowledgeExtractor(
            model_name=config.model_name,
            temperature=config.temperature,
            api_key=api_key
        )
        
        # Create pipelines with configured domain extractor
        pipeline_no_domain = Pipeline(config)
        pipeline_no_domain.domain_extractor = domain_extractor
        result_no_domain = pipeline_no_domain.run(
            data=data,
            target_column='mpg',
            problem_description="Regression problem predicting miles per gallon (MPG) of automobiles",
            is_classification=False
        )
        
        pipeline_with_domain = Pipeline(config)
        pipeline_with_domain.domain_extractor = domain_extractor
        result_with_domain = pipeline_with_domain.run(
            data=data,
            target_column='mpg',
            problem_description="""
            Regression problem predicting miles per gallon (MPG) of automobiles.
            Features:
            - cylinders: Number of cylinders in the engine (integer)
            - displacement: Engine displacement in cubic inches (continuous)
            - horsepower: Engine horsepower (continuous)
            - weight: Vehicle weight in pounds (continuous)
            - acceleration: Time to accelerate from 0 to 60 mph in seconds (continuous)
            - model_year: Vehicle model year (integer, 70-82 representing 1970-1982)
            - origin: Origin of car (categorical: american, european, asian)
            
            Domain context:
            - Fuel efficiency (MPG) generally decreases with vehicle weight
            - Higher horsepower engines typically consume more fuel
            - Larger engine displacement generally correlates with lower MPG
            - Cars with more cylinders typically have lower fuel efficiency
            - Technological improvements over model years generally increased fuel efficiency
            - There are tradeoffs between performance (acceleration) and fuel efficiency
            - Different regions (origins) had different emission standards and design philosophies
            - The dataset contains cars from the 1970s and early 1980s during fuel crises
            """,
            is_classification=False
        )
    
    # Get transformed features from the pipeline results
    print("\nAvailable attributes in result_no_domain:", dir(result_no_domain))
    
    # Get the transformed data
    no_domain_features = result_no_domain.transformed_data
    with_domain_features = result_with_domain.transformed_data
    
    # Drop the target column if it exists in the transformed data
    if 'mpg' in no_domain_features.columns:
        no_domain_features = no_domain_features.drop('mpg', axis=1)
    if 'mpg' in with_domain_features.columns:
        with_domain_features = with_domain_features.drop('mpg', axis=1)
    
    # Print feature information for debugging
    print("\nNo Domain Features Shape:", no_domain_features.shape)
    print("With Domain Features Shape:", with_domain_features.shape)
    print("\nBaseline X Columns:")
    print(X.columns.tolist())
    print("\nFASTER (No Domain Knowledge) X Columns:")
    print(no_domain_features.columns.tolist())
    print("\nFASTER (With Domain Knowledge) X Columns:")
    print(with_domain_features.columns.tolist())
    
    # Evaluate FASTER-enhanced models
    faster_no_domain_scores = evaluate_model(
        no_domain_features,
        y
    )
    
    faster_with_domain_scores = evaluate_model(
        with_domain_features,
        y
    )
    
    # Print results
    print(f"\nModel Performance Comparison ({'Mock' if use_mock else 'Real'} LLM):")
    
    print("\nBaseline Model:")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in baseline_scores.items() if k.startswith('train')]:
        print(f"    {metric.replace('train_', '')}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in baseline_scores.items() if k.startswith('test')]:
        print(f"    {metric.replace('test_', '')}: {score:.4f}")
    
    print("\nFASTER (No Domain Knowledge):")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in faster_no_domain_scores.items() if k.startswith('train')]:
        print(f"    {metric.replace('train_', '')}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in faster_no_domain_scores.items() if k.startswith('test')]:
        print(f"    {metric.replace('test_', '')}: {score:.4f}")
    
    print("\nFASTER (With Domain Knowledge):")
    print("  Train Metrics:")
    for metric, score in [(k, v) for k, v in faster_with_domain_scores.items() if k.startswith('train')]:
        print(f"    {metric.replace('train_', '')}: {score:.4f}")
    print("  Test Metrics:")
    for metric, score in [(k, v) for k, v in faster_with_domain_scores.items() if k.startswith('test')]:
        print(f"    {metric.replace('test_', '')}: {score:.4f}")
    
    # Print generated features
    print("\nGenerated Features:")
    print("\nNo Domain Knowledge:")
    print(result_no_domain.selected_features)
    print("\nWith Domain Knowledge:")
    print(result_with_domain.selected_features)
    
    # Assertions to ensure FASTER is working
    assert result_no_domain.selected_features is not None
    assert result_with_domain.selected_features is not None
    assert len(result_no_domain.selected_features) > 0
    assert len(result_with_domain.selected_features) > 0
    
    all_scores = {
        'baseline': baseline_scores,
        'faster_no_domain': faster_no_domain_scores,
        'faster_with_domain': faster_with_domain_scores
    }
    
    print("\nAll performance metrics:", all_scores) 