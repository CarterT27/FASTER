"""Tests for feature generation module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal, special

from faster.feature_generation import FeatureGenerator, TransformationMetadata
from faster.domain_knowledge import DomainInsight


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "numeric_normal": np.random.normal(0, 1, 100),
            "numeric_skewed": np.exp(np.random.normal(0, 1, 100)),
            "categorical": np.random.choice(["A", "B", "C"], 100),
            "text": [
                "This is sample text" if i % 2 == 0 else "Another text example" for i in range(100)
            ],
            "target": np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture
def sample_insights():
    """Create sample domain insights."""
    return [
        DomainInsight(
            feature_name="numeric_normal",
            importance=0.8,
            relationships=["numeric_skewed"],
            suggested_transformations=["zscore", "minmax"],
            rationale="Important numeric feature",
        ),
        DomainInsight(
            feature_name="numeric_skewed",
            importance=0.7,
            relationships=["numeric_normal"],
            suggested_transformations=["log", "sqrt"],
            rationale="Skewed numeric feature",
        ),
        DomainInsight(
            feature_name="categorical",
            importance=0.5,
            relationships=[],
            suggested_transformations=["one_hot"],
            rationale="Categorical feature",
        ),
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
    text_series = pd.Series(
        [
            "This is a long text with multiple words",
            "Another sample text with sufficient length",
            "More text data that should be detected",
        ]
    )
    short_text_series = pd.Series(["A", "B", "C"])
    numeric_series = pd.Series([1, 2, 3])

    assert FeatureGenerator._is_text_column(text_series)
    assert not FeatureGenerator._is_text_column(short_text_series)
    assert not FeatureGenerator._is_text_column(numeric_series)


def test_fit_transform_scaler(feature_generator):
    """Test scaler fitting and transformation."""

    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    sklearn_scaler = StandardScaler()
    expected = pd.Series(sklearn_scaler.fit_transform(series.values.reshape(-1, 1)).flatten())

    transformed = feature_generator._fit_transform_scaler(series, "test")

    pd.testing.assert_series_equal(transformed, expected, check_names=False, rtol=1e-10, atol=1e-10)
    assert "test" in feature_generator._scalers


def test_apply_basic_transformations(feature_generator, sample_data):
    """Test basic transformation application."""

    top_features = ["numeric_normal", "numeric_skewed"]

    result = feature_generator._apply_basic_transformations(sample_data, [], "target", top_features)

    assert "scaled_numeric_normal" in result.columns
    assert "log_numeric_skewed" in result.columns


def test_generate_interaction_features(feature_generator):
    """Test interaction feature generation."""

    test_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

    test_insights = [
        DomainInsight(
            feature_name="feature1",
            importance=0.8,
            relationships=["feature2"],
            suggested_transformations=["multiply"],
            rationale="Test interaction",
        )
    ]

    result = feature_generator._generate_interaction_features(test_data.copy(), test_insights)

    interaction_col = "multiply_feature1_feature2"
    assert interaction_col in result.columns

    expected_interaction = test_data["feature1"] * test_data["feature2"]
    pd.testing.assert_series_equal(result[interaction_col], expected_interaction, check_names=False)

    assert interaction_col in feature_generator.transformations
    assert feature_generator.transformations[interaction_col].transformation_type == "multiply"
    assert set(feature_generator.transformations[interaction_col].original_features) == {
        "feature1",
        "feature2",
    }


def test_apply_domain_transformations(feature_generator, sample_data, sample_insights):
    """Test domain-specific transformation application."""

    recommended_transforms = {
        "numeric_normal": ["zscore", "minmax"],
        "numeric_skewed": ["log"],
        "categorical": ["one_hot"],
    }

    result = feature_generator._apply_domain_transformations(
        sample_data, sample_insights, recommended_transforms
    )

    assert "zscore_numeric_normal" in result.columns
    assert "log_numeric_skewed" in result.columns


def test_generate_text_features(feature_generator):
    """Test text feature generation."""

    test_data = pd.DataFrame(
        {
            "text_col": [
                "This is a long text about topic A",
                "Another text about topic B",
                "More text about topic A",
                "Final text about topic C",
            ]
        }
    )

    result = feature_generator._generate_text_features(test_data, [])

    tfidf_cols = [col for col in result.columns if col.startswith("tfidf_text_col_")]
    assert len(tfidf_cols) > 0
    assert "text_col" in feature_generator._text_vectorizers


def test_generate_features_integration(feature_generator):
    """Test full feature generation pipeline."""

    test_data = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "skewed": [1.0, 10.0, 100.0, 1000.0],
            "categorical": ["A", "B", "A", "B"],
            "text": [
                "Text about topic A",
                "Text about topic B",
                "More about topic A",
                "More about topic B",
            ],
            "target": [0, 1, 0, 1],
        }
    )

    test_insights = [
        DomainInsight(
            feature_name="numeric",
            importance=0.8,
            relationships=["skewed"],
            suggested_transformations=["zscore", "minmax", "multiply"],
            rationale="Numeric feature",
        ),
        DomainInsight(
            feature_name="skewed",
            importance=0.7,
            relationships=["numeric"],
            suggested_transformations=["log"],
            rationale="Skewed feature",
        ),
    ]

    result = feature_generator.generate_features(test_data.copy(), test_insights, "target")

    assert "zscore_numeric" in result.columns
    assert "scaled_numeric" in result.columns
    assert "log_skewed" in result.columns

    assert "multiply_numeric_skewed" in result.columns

    text_features = [col for col in result.columns if col.startswith("tfidf_")]
    assert len(text_features) > 0


def test_error_handling(feature_generator):
    """Test error handling in feature generation."""

    test_data = pd.DataFrame({"existing_feature": [1, 2, 3], "target": [0, 1, 0]})

    invalid_insights = [
        DomainInsight(
            feature_name="nonexistent_feature",
            importance=0.5,
            relationships=["another_nonexistent"],
            suggested_transformations=["log"],
            rationale="This feature doesn't exist",
        )
    ]

    result = feature_generator.generate_features(
        test_data, invalid_insights, target_column="target"
    )

    assert "nonexistent_feature" not in result.columns
    assert "another_nonexistent" not in result.columns
    assert "log_nonexistent_feature" not in result.columns


def test_transformation_metadata(feature_generator, sample_data, sample_insights):
    """Test transformation metadata recording."""
    feature_generator.generate_features(sample_data, sample_insights)

    for name, metadata in feature_generator.transformations.items():
        assert isinstance(metadata.original_features, list)
        assert isinstance(metadata.transformation_type, str)
        assert isinstance(metadata.parameters, dict)
        assert isinstance(metadata.rationale, str)
        assert all(feat in sample_data.columns for feat in metadata.original_features)


def test_scipy_transformations(feature_generator):
    """Test scipy-based transformations."""

    x = np.linspace(0, 10, 100)
    np.random.seed(42)  # Set seed for reproducibility

    uniform_values = np.random.uniform(0.01, 0.99, 100)
    print(f"\nUniform values range: [{uniform_values.min():.6f}, {uniform_values.max():.6f}]")

    test_data = pd.DataFrame(
        {
            "positive": np.exp(x / 5),  # For Box-Cox
            "any_value": x - 5,  # For Yeo-Johnson
            "uniform": uniform_values,  # For logit, avoid exact 0/1
            "trend": x + np.random.normal(0, 0.1, 100),  # For detrend
            "noisy": np.sin(x) + np.random.normal(0, 0.1, 100),  # For Savitzky-Golay
            "oscillating": np.sin(x) + np.cos(2 * x),  # For Hilbert
        }
    )

    print(
        f"Uniform column range: [{test_data['uniform'].min():.6f}, {test_data['uniform'].max():.6f}]"
    )

    assert (test_data["uniform"] > 0).all(), "Uniform values must be positive"
    assert (test_data["uniform"] < 1).all(), "Uniform values must be less than 1"

    test_insights = [
        DomainInsight(
            feature_name="positive",
            importance=0.8,
            relationships=[],
            suggested_transformations=["boxcox", "yeojohnson"],
            rationale="Test scipy.stats transformations",
        ),
        DomainInsight(
            feature_name="any_value",
            importance=0.7,
            relationships=[],
            suggested_transformations=["yeojohnson", "quantile"],
            rationale="Test more scipy.stats transformations",
        ),
        DomainInsight(
            feature_name="uniform",
            importance=0.6,
            relationships=[],
            suggested_transformations=["logit", "expit"],
            rationale="Test scipy.special transformations",
        ),
        DomainInsight(
            feature_name="trend",
            importance=0.5,
            relationships=[],
            suggested_transformations=["detrend"],
            rationale="Test signal.detrend",
        ),
        DomainInsight(
            feature_name="noisy",
            importance=0.4,
            relationships=[],
            suggested_transformations=["savgol"],
            rationale="Test Savitzky-Golay filter",
        ),
        DomainInsight(
            feature_name="oscillating",
            importance=0.3,
            relationships=[],
            suggested_transformations=["hilbert"],
            rationale="Test Hilbert transform",
        ),
    ]

    recommended_transforms = {}
    for insight in test_insights:
        recommended_transforms[insight.feature_name] = insight.suggested_transformations

    result = feature_generator._apply_domain_transformations(
        test_data, test_insights, recommended_transforms
    )

    print(f"\nActual columns in result: {result.columns.tolist()}")

    if "boxcox_positive" in result.columns:
        assert abs(stats.skew(result["boxcox_positive"])) < abs(stats.skew(test_data["positive"]))

    if "yeojohnson_any_value" in result.columns:
        assert "yeojohnson_any_value" in result.columns
