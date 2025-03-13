"""Tests for feature generation module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from sklearn.preprocessing import StandardScaler

from faster.feature_generation import FeatureGenerator, TransformationMetadata
from faster.domain_knowledge import DomainInsight

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric_normal': np.random.normal(0, 1, 100),
        'numeric_skewed': np.exp(np.random.normal(0, 1, 100)),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'text': [
            'This is sample text' if i % 2 == 0 else 'Another text example'
            for i in range(100)
        ],
        'target': np.random.randint(0, 2, 100)
    })

@pytest.fixture
def sample_insights():
    """Create sample domain insights."""
    return [
        DomainInsight(
            feature_name="numeric_normal",
            importance=0.8,
            relationships=["numeric_skewed"],
            suggested_transformations=["zscore", "minmax"],
            rationale="Important numeric feature"
        ),
        DomainInsight(
            feature_name="numeric_skewed",
            importance=0.7,
            relationships=["numeric_normal"],
            suggested_transformations=["log", "sqrt"],
            rationale="Skewed numeric feature"
        ),
        DomainInsight(
            feature_name="categorical",
            importance=0.5,
            relationships=[],
            suggested_transformations=["one_hot"],
            rationale="Categorical feature"
        )
    ]

@pytest.fixture
def feature_generator():
    """Create a FeatureGenerator instance."""
    return FeatureGenerator()

def test_init(feature_generator):
    """Test initialization."""
    assert isinstance(feature_generator.transformations, dict)
    assert isinstance(feature_generator._scalers, dict)
    assert isinstance(feature_generator._text_vectorizers, dict)

def test_should_apply_log_transform():
    """Test log transform detection."""
    normal_series = pd.Series(np.random.normal(0, 1, 1000))
    skewed_series = pd.Series(np.exp(np.random.normal(0, 1, 1000)))
    
    assert not FeatureGenerator._should_apply_log_transform(normal_series)
    assert FeatureGenerator._should_apply_log_transform(skewed_series)

def test_is_text_column():
    """Test text column detection."""
    text_series = pd.Series([
        'This is a long text with multiple words',
        'Another sample text with sufficient length',
        'More text data that should be detected'
    ])
    short_text_series = pd.Series(['A', 'B', 'C'])
    numeric_series = pd.Series([1, 2, 3])
    
    assert FeatureGenerator._is_text_column(text_series)
    assert not FeatureGenerator._is_text_column(short_text_series)
    assert not FeatureGenerator._is_text_column(numeric_series)

def test_fit_transform_scaler(feature_generator):
    """Test scaler fitting and transformation."""
    # Create a simple series with known values
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Use sklearn's StandardScaler for comparison
    sklearn_scaler = StandardScaler()
    expected = pd.Series(sklearn_scaler.fit_transform(series.values.reshape(-1, 1)).flatten())
    
    # Test our implementation
    transformed = feature_generator._fit_transform_scaler(series, 'test')
    
    # Compare with sklearn's output
    pd.testing.assert_series_equal(
        transformed,
        expected,
        check_names=False,
        rtol=1e-10,
        atol=1e-10
    )
    assert 'test' in feature_generator._scalers

def test_apply_basic_transformations(feature_generator, sample_data):
    """Test basic transformation application."""
    result = feature_generator._apply_basic_transformations(
        sample_data,
        [],
        'target'
    )
    
    assert 'scaled_numeric_normal' in result.columns
    assert 'scaled_numeric_skewed' in result.columns
    assert 'log_numeric_skewed' in result.columns
    assert len(feature_generator.transformations) >= 3

def test_generate_interaction_features(feature_generator):
    """Test interaction feature generation."""
    # Create simple test data
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    
    test_insights = [
        DomainInsight(
            feature_name="feature1",
            importance=0.8,
            relationships=["feature2"],
            suggested_transformations=["multiply"],
            rationale="Test interaction"
        )
    ]
    
    # Generate interaction features
    result = feature_generator._generate_interaction_features(test_data.copy(), test_insights)
    
    # Check if interaction column was created
    interaction_col = 'multiply_feature1_feature2'
    assert interaction_col in result.columns
    
    # Verify interaction values
    expected_interaction = test_data['feature1'] * test_data['feature2']
    pd.testing.assert_series_equal(result[interaction_col], expected_interaction, check_names=False)
    
    # Check transformation metadata
    assert interaction_col in feature_generator.transformations
    assert feature_generator.transformations[interaction_col].transformation_type == 'multiply'
    assert set(feature_generator.transformations[interaction_col].original_features) == {'feature1', 'feature2'}

def test_apply_domain_transformations(feature_generator, sample_data, sample_insights):
    """Test domain-specific transformation application."""
    result = feature_generator._apply_domain_transformations(
        sample_data,
        sample_insights
    )
    
    assert 'zscore_numeric_normal' in result.columns
    assert 'minmax_numeric_normal' in result.columns
    assert 'log_numeric_skewed' in result.columns
    assert 'sqrt_numeric_skewed' in result.columns

def test_generate_text_features(feature_generator):
    """Test text feature generation."""
    # Create test data with clear text features
    test_data = pd.DataFrame({
        'text_col': [
            'This is a long text about topic A',
            'Another text about topic B',
            'More text about topic A',
            'Final text about topic C'
        ]
    })
    
    result = feature_generator._generate_text_features(test_data, [])
    
    # Check if TF-IDF features were created
    tfidf_cols = [col for col in result.columns if col.startswith('tfidf_text_col_')]
    assert len(tfidf_cols) > 0
    assert 'text_col' in feature_generator._text_vectorizers

def test_generate_features_integration(feature_generator):
    """Test full feature generation pipeline."""
    # Create test data with clear features for transformation
    test_data = pd.DataFrame({
        'numeric': [1.0, 2.0, 3.0, 4.0],
        'skewed': [1.0, 10.0, 100.0, 1000.0],
        'categorical': ['A', 'B', 'A', 'B'],
        'text': [
            'Text about topic A',
            'Text about topic B',
            'More about topic A',
            'More about topic B'
        ],
        'target': [0, 1, 0, 1]
    })
    
    test_insights = [
        DomainInsight(
            feature_name="numeric",
            importance=0.8,
            relationships=["skewed"],
            suggested_transformations=["zscore", "minmax", "multiply"],
            rationale="Numeric feature"
        ),
        DomainInsight(
            feature_name="skewed",
            importance=0.7,
            relationships=["numeric"],
            suggested_transformations=["log"],
            rationale="Skewed feature"
        )
    ]
    
    result = feature_generator.generate_features(
        test_data.copy(),
        test_insights,
        'target'
    )
    
    # Check basic transformations
    assert 'zscore_numeric' in result.columns
    assert 'minmax_numeric' in result.columns
    assert 'log_skewed' in result.columns
    
    # Check interaction features
    assert 'multiply_numeric_skewed' in result.columns
    expected_interaction = test_data['numeric'] * test_data['skewed']
    pd.testing.assert_series_equal(result['multiply_numeric_skewed'], expected_interaction, check_names=False)
    
    # Check text features
    assert any(col.startswith('tfidf_text_') for col in result.columns)
    
    # Verify all transformations are tracked
    assert all(col in feature_generator.transformations for col in result.columns if col not in test_data.columns)

def test_error_handling(feature_generator):
    """Test error handling in feature generation."""
    # Create minimal test data
    test_data = pd.DataFrame({
        'existing_feature': [1, 2, 3],
        'target': [0, 1, 0]
    })
    
    invalid_insights = [
        DomainInsight(
            feature_name="nonexistent_feature",
            importance=0.5,
            relationships=["another_nonexistent"],
            suggested_transformations=["log"],
            rationale="This feature doesn't exist"
        )
    ]
    
    # Should not raise exception but log warning
    result = feature_generator.generate_features(
        test_data,
        invalid_insights,
        target_column='target'
    )
    
    # Should only have original columns plus any automatic transformations of existing numeric columns
    assert 'nonexistent_feature' not in result.columns
    assert 'another_nonexistent' not in result.columns
    assert 'log_nonexistent_feature' not in result.columns

def test_transformation_metadata(feature_generator, sample_data, sample_insights):
    """Test transformation metadata recording."""
    feature_generator.generate_features(sample_data, sample_insights)
    
    for name, metadata in feature_generator.transformations.items():
        assert isinstance(metadata.original_features, list)
        assert isinstance(metadata.transformation_type, str)
        assert isinstance(metadata.parameters, dict)
        assert isinstance(metadata.rationale, str)
        assert all(feat in sample_data.columns for feat in metadata.original_features) 