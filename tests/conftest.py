"""Shared test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import logging
import os

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""



    yield


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

def colored_metric_output(metric_name: str, model_score: float, baseline_score: float) -> str:
    """
    Format metric output with color based on performance compared to baseline.
    
    Args:
        metric_name: Name of the metric (e.g., 'r2', 'accuracy')
        model_score: Score achieved by the model being evaluated
        baseline_score: Score achieved by the baseline model
        
    Returns:
        Formatted string with appropriate color (green for better, red for worse)
    """

    GREEN = '\033[92m'  # Bright green
    RED = '\033[91m'    # Bright red
    RESET = '\033[0m'   # Reset to default

    error_metrics = ['rmse', 'mae', 'error', 'loss']
    is_error_metric = any(err in metric_name.lower() for err in error_metrics)

    if is_error_metric:
        is_better = model_score < baseline_score
    else:
        is_better = model_score > baseline_score

    diff = model_score - baseline_score
    diff_str = f"{diff:.4f}"
    if not is_error_metric:
        diff_str = f"+{diff_str}" if diff > 0 else f"{diff_str}"
    else:
        diff_str = f"{diff_str}" if diff > 0 else f"{diff_str}"

    color = GREEN if is_better else RED
    
    return f"{metric_name}: {color}{model_score:.4f} ({diff_str}){RESET}" 