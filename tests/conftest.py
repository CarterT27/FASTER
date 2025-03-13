"""Shared test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import logging
import os

# Configure logging for tests
@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Ensure test environment is properly set up
@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Only set a default value if the environment variable doesn't exist
    if 'OPENROUTER_API_KEY' not in os.environ:
        os.environ['OPENROUTER_API_KEY'] = 'dummy_key_for_testing'
    
    # Store the original value to restore later
    original_api_key = os.environ.get('OPENROUTER_API_KEY')
    
    # Clean up after tests
    yield
    
    # Restore the original value if it existed
    if original_api_key:
        os.environ['OPENROUTER_API_KEY'] = original_api_key
    elif 'OPENROUTER_API_KEY' in os.environ:
        del os.environ['OPENROUTER_API_KEY']

@pytest.fixture
def numeric_data():
    """Create sample numeric data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 50),
        'feature2': np.random.exponential(2, 50),
        'target': np.random.randint(0, 2, 50)
    })

@pytest.fixture
def mixed_data():
    """Create sample mixed-type data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric': np.random.normal(0, 1, 50),
        'categorical': np.random.choice(['A', 'B', 'C'], 50),
        'text': [f'Sample text {i}' for i in range(50)],
        'target': np.random.randint(0, 2, 50)
    })

@pytest.fixture
def sample_domain_context():
    """Provide sample domain context for testing."""
    return {
        'problem_type': 'binary_classification',
        'domain': 'customer_churn',
        'target_description': 'Customer churned (1) or not (0)',
        'feature_descriptions': {
            'numeric': 'Customer satisfaction score',
            'categorical': 'Customer segment',
            'text': 'Customer feedback'
        }
    } 