"""Integration test using the Titanic dataset to compare FASTER with baseline models."""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import warnings
from typing import Dict, List, Tuple
from unittest.mock import patch
import os
import openai
import logging

from faster.pipeline import Pipeline, PipelineConfig
from faster.feature_selection import SelectionCriteria
from faster.domain_knowledge import DomainKnowledgeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock LLM response for Titanic dataset
MOCK_LLM_RESPONSE = [
    {
        "feature_name": "Sex",
        "importance": 0.9,
        "relationships": ["Age", "Pclass"],
        "suggested_transformations": ["one_hot", "label"],
        "rationale": "Gender was a primary factor in survival due to 'women and children first' policy"
    },
    {
        "feature_name": "Age",
        "importance": 0.8,
        "relationships": ["Sex", "Pclass"],
        "suggested_transformations": ["log", "zscore", "binning"],
        "rationale": "Age affected survival chances with children having priority"
    },
    {
        "feature_name": "Pclass",
        "importance": 0.7,
        "relationships": ["Fare"],
        "suggested_transformations": ["one_hot", "ordinal"],
        "rationale": "Passenger class correlated with survival due to cabin location and access to lifeboats"
    },
    {
        "feature_name": "Fare",
        "importance": 0.6,
        "relationships": ["Pclass"],
        "suggested_transformations": ["log", "zscore"],
        "rationale": "Ticket fare indicates passenger class and potentially better access to survival"
    },
    {
        "feature_name": "SibSp",
        "importance": 0.5,
        "relationships": ["Parch"],
        "suggested_transformations": ["zscore", "interaction"],
        "rationale": "Family size affected survival chances"
    }
]

def load_and_preprocess_titanic() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Titanic dataset."""
    # Load Titanic dataset
    titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    
    # Basic feature selection
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = titanic[features].copy()
    y = titanic['Survived']
    
    # Handle missing values
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Sex', 'Embarked']
    
    # Impute numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    
    # Impute categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    
    # Encode categorical features
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
    
    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    return X, y

def evaluate_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate model using cross-validation."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Perform cross-validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            clf, X, y,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )
    
    # Calculate mean scores
    return {
        'accuracy': cv_results['test_accuracy'].mean(),
        'f1': cv_results['test_f1'].mean(),
        'roc_auc': cv_results['test_roc_auc'].mean()
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
    # Ensure key is properly formatted for Authorization header
    if not api_key.startswith("Bearer ") and not api_key.startswith("bearer "):
        auth_header = f"Bearer {api_key}"
    else:
        auth_header = api_key
        
    logger.info(f"Configuring OpenAI client with base URL: https://openrouter.ai/api/v1")
    
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/CarterT27/FASTER",  # Project repository
            "X-Title": "FASTER Integration Tests",
        }
    )

@pytest.mark.parametrize("use_mock", [True, False])
def test_titanic_integration(use_mock):
    """Integration test comparing baseline and FASTER-enhanced models on Titanic dataset."""
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
    X, y = load_and_preprocess_titanic()
    data = X.copy()
    data['Survived'] = y
    
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
                target_column='Survived',
                problem_description="Binary classification problem predicting survival",
                is_classification=True
            )
            
            pipeline_with_domain = Pipeline(config)
            result_with_domain = pipeline_with_domain.run(
                data=data,
                target_column='Survived',
                problem_description="""
                Binary classification problem predicting passenger survival on the Titanic.
                Features:
                - Pclass: Ticket class (1st, 2nd, 3rd)
                - Sex: Passenger's sex
                - Age: Passenger's age
                - SibSp: Number of siblings/spouses aboard
                - Parch: Number of parents/children aboard
                - Fare: Passenger fare
                - Embarked: Port of embarkation
                
                Domain context:
                - First class passengers had higher survival rates
                - Women and children had priority in lifeboats
                - Families traveling together might have different survival patterns
                - Ticket fare could indicate wealth and access to better areas of the ship
                - Port of embarkation could indicate passenger's social status
                """,
                is_classification=True
            )
    else:
        # Create domain knowledge extractor with OpenRouter configuration
        domain_extractor = DomainKnowledgeExtractor(
            model_name=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            openai_client=client
        )
        
        # Create pipelines with configured domain extractor
        pipeline_no_domain = Pipeline(config)
        pipeline_no_domain.domain_extractor = domain_extractor
        result_no_domain = pipeline_no_domain.run(
            data=data,
            target_column='Survived',
            problem_description="Binary classification problem predicting survival",
            is_classification=True
        )
        
        pipeline_with_domain = Pipeline(config)
        pipeline_with_domain.domain_extractor = domain_extractor
        result_with_domain = pipeline_with_domain.run(
            data=data,
            target_column='Survived',
            problem_description="""
            Binary classification problem predicting passenger survival on the Titanic.
            Features:
            - Pclass: Ticket class (1st, 2nd, 3rd)
            - Sex: Passenger's sex
            - Age: Passenger's age
            - SibSp: Number of siblings/spouses aboard
            - Parch: Number of parents/children aboard
            - Fare: Passenger fare
            - Embarked: Port of embarkation
            
            Domain context:
            - First class passengers had higher survival rates
            - Women and children had priority in lifeboats
            - Families traveling together might have different survival patterns
            - Ticket fare could indicate wealth and access to better areas of the ship
            - Port of embarkation could indicate passenger's social status
            """,
            is_classification=True
        )
    
    # Get transformed features from the pipeline results
    print("\nAvailable attributes in result_no_domain:", dir(result_no_domain))
    
    # Get the transformed data
    no_domain_features = result_no_domain.transformed_data
    with_domain_features = result_with_domain.transformed_data
    
    # Drop the target column if it exists in the transformed data
    if 'Survived' in no_domain_features.columns:
        no_domain_features = no_domain_features.drop('Survived', axis=1)
    if 'Survived' in with_domain_features.columns:
        with_domain_features = with_domain_features.drop('Survived', axis=1)
    
    # Print feature information for debugging
    print("\nNo Domain Features Shape:", no_domain_features.shape)
    print("With Domain Features Shape:", with_domain_features.shape)
    print("\nNo Domain Features Columns:", no_domain_features.columns.tolist())
    
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
    for metric, score in baseline_scores.items():
        print(f"{metric}: {score:.4f}")
    
    print("\nFASTER (No Domain Knowledge):")
    for metric, score in faster_no_domain_scores.items():
        print(f"{metric}: {score:.4f}")
    
    print("\nFASTER (With Domain Knowledge):")
    for metric, score in faster_with_domain_scores.items():
        print(f"{metric}: {score:.4f}")
    
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