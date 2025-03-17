"""Tests for pipeline module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os

from faster.pipeline import Pipeline, PipelineConfig
from faster.feature_selection import SelectionCriteria
from faster.domain_knowledge import DomainInsight
from faster.statistical_evaluation import FeatureStatistics

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_0': np.random.normal(0, 1, 100),
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(0, 1, 100),
        'feature_3': np.random.normal(0, 1, 100),
        'feature_4': np.random.normal(0, 1, 100),
        'target': np.random.randint(0, 2, 100)
    })

@pytest.fixture
def pipeline_config():
    """Create pipeline configuration."""
    return PipelineConfig(
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

def test_pipeline_initialization(pipeline_config):
    """Test pipeline initialization."""
    with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'dummy_key'}):
        pipeline = Pipeline(pipeline_config)
        assert pipeline.config == pipeline_config
        assert pipeline.domain_extractor is not None
        assert pipeline.feature_generator is not None
        assert pipeline.statistical_evaluator is not None
        assert pipeline.feature_selector is not None

@patch('faster.domain_knowledge.DomainKnowledgeExtractor._query_llm_with_retry')
def test_pipeline_run(mock_query, sample_data, pipeline_config):
    """Test pipeline execution."""

    mock_query.return_value = [
        {
            "feature_name": "feature_0",
            "importance": 0.8,
            "relationships": ["feature_1"],
            "suggested_transformations": ["zscore", "multiply"],
            "rationale": "Test feature"
        }
    ]

    with patch('faster.statistical_evaluation.StatisticalEvaluator.evaluate_features') as mock_stats:
        mock_stats.return_value = [
            FeatureStatistics(
                feature_name="feature_0",
                correlation=0.8,
                p_value=0.01,
                effect_size=0.5,
                mutual_information=0.3,
                test_method="pearson",
                assumptions_met={"normality": True},
                warnings=[]
            )
        ]
        
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'dummy_key'}):
            pipeline = Pipeline(pipeline_config)
            result = pipeline.run(
                data=sample_data,
                target_column="target",
                problem_description="Binary classification problem with synthetic data",
                is_classification=True,
            )
    
    assert result.selected_features is not None
    assert len(result.selected_features) > 0
    assert result.feature_metadata is not None
    assert result.performance_metrics is not None

def test_pipeline_validation(pipeline_config):
    """Test pipeline validation."""
    with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'dummy_key'}):
        pipeline = Pipeline(pipeline_config)
        
        with pytest.raises(ValueError, match="Data cannot be None"):
            pipeline.run(
                data=None,
                target_column="target",
                problem_description="Test problem",
            )
        
        with pytest.raises(ValueError, match="Target column cannot be empty"):
            pipeline.run(
                data=pd.DataFrame({'a': [1, 2, 3]}),
                target_column="",
                problem_description="Test problem",
            )
        
        with pytest.raises(ValueError, match="Problem description cannot be empty"):
            pipeline.run(
                data=pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}),
                target_column="a",
                problem_description="",
            ) 