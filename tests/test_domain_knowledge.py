"""Tests for domain knowledge extraction module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import json
import os
import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from pydantic import ValidationError

from faster.domain_knowledge import DomainKnowledgeExtractor, DomainInsight, PromptConfig

@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric_feature': [1, 2, 3, 4, 5],
        'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
        'text_feature': [
            'This is a long text',
            'Another text sample',
            'More text data here',
            'Sample text content',
            'Final text example'
        ],
        'target': [0, 1, 0, 1, 1]
    })

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return json.dumps([
        {
            "feature_name": "numeric_feature",
            "importance": 0.8,
            "relationships": ["categorical_feature"],
            "suggested_transformations": ["log", "zscore"],
            "rationale": "High importance numeric feature with right-skewed distribution"
        },
        {
            "feature_name": "categorical_feature",
            "importance": 0.6,
            "relationships": ["numeric_feature"],
            "suggested_transformations": ["one_hot"],
            "rationale": "Categorical feature with moderate correlation to target"
        }
    ])

@pytest.fixture
def domain_extractor():
    """Create a DomainKnowledgeExtractor instance."""
    with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'dummy_key'}):
        # Use a mock to avoid actual API calls
        with patch('openai.OpenAI') as mock_openai:
            # Create a mock client with appropriate attributes
            mock_client = Mock()
            # Set api_key to a string to avoid TypeError in _query_llm_with_retry
            mock_client.api_key = ""
            # Set base_url to a string to avoid attribute errors
            mock_client.base_url = "https://openrouter.ai/api/v1"
            mock_openai.return_value = mock_client
            
            extractor = DomainKnowledgeExtractor()
            # Replace the client to avoid actual API calls
            extractor.client = mock_client
            return extractor

def test_init_without_api_key():
    """Test initialization without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY.*required"):
            DomainKnowledgeExtractor()

def test_init_with_custom_config(domain_extractor):
    """Test initialization with custom prompt config."""
    custom_config = PromptConfig(
        context_template="custom context",
        expert_template="custom expert",
        feature_suggestion_template="custom suggestion",
        validation_template="custom validation"
    )
    extractor = DomainKnowledgeExtractor(prompt_config=custom_config)
    assert extractor.prompt_config.context_template == "custom context"

def test_parse_llm_response_valid_json(domain_extractor, mock_llm_response):
    """Test parsing valid JSON LLM response."""
    insights = domain_extractor._parse_llm_response(mock_llm_response)
    assert len(insights) == 2
    assert insights[0]["feature_name"] == "numeric_feature"
    assert isinstance(insights[0]["importance"], float)

def test_parse_llm_response_invalid_json(domain_extractor):
    """Test parsing invalid JSON LLM response."""
    invalid_response = "Not a JSON response"
    with pytest.raises(ValueError, match="Could not parse LLM response"):
        domain_extractor._parse_llm_response(invalid_response)

def test_parse_llm_response_missing_fields(domain_extractor):
    """Test parsing response with missing required fields."""
    invalid_json = json.dumps([{
        "feature_name": "test",
        "importance": 0.5
        # Missing relationships, suggested_transformations, and rationale
    }])
    
    with pytest.raises(ValueError, match="Missing required fields"):
        insights = domain_extractor._parse_llm_response(invalid_json)
        # The validation should fail before we get here
        pytest.fail("Expected ValueError but no exception was raised")

def test_generate_data_summary(domain_extractor, sample_data):
    """Test data summary generation."""
    summary = domain_extractor._generate_data_summary(sample_data, "target")
    assert summary["n_samples"] == 5
    assert summary["n_features"] == 3
    assert "feature_types" in summary
    assert "missing_values" in summary
    assert "target_distribution" in summary

@patch('openai.OpenAI')
def test_query_llm_with_retry_success(mock_openai, domain_extractor, mock_llm_response):
    """Test successful LLM query with retry."""
    # Create a proper mock response
    mock_message = ChatCompletionMessage(
        content=mock_llm_response,
        role="assistant"
    )
    mock_completion = ChatCompletion(
        id="test_id",
        choices=[{
            "finish_reason": "stop",
            "index": 0,
            "message": mock_message
        }],
        created=1234567890,
        model="test-model",
        object="chat.completion"
    )
    
    # Configure the mock client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_completion
    mock_openai.return_value = mock_client
    
    # Replace the client in domain_extractor
    domain_extractor.client = mock_client
    
    insights = domain_extractor._query_llm_with_retry("test prompt")
    assert len(insights) == 2
    assert mock_client.chat.completions.create.call_count == 1
    
    # Verify correct API call including expected headers
    mock_client.chat.completions.create.assert_called_with(
        model="deepseek/deepseek-chat:free",
        messages=[{"role": "user", "content": "test prompt"}],
        temperature=0.0,
        extra_headers={
            "HTTP-Referer": "https://github.com/cartertran/faster",
            "X-Title": "FASTER Feature Selection Tool"
        }
    )

@patch('openai.OpenAI')
def test_query_llm_with_retry_failure(mock_openai, domain_extractor):
    """Test LLM query with retry on failure."""
    # Configure mock to raise authentication error
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        httpx.HTTPStatusError(
            "401 Unauthorized",
            request=Mock(),
            response=Mock(status_code=401)
        )
    ] * 3  # Will raise error 3 times
    
    mock_openai.return_value = mock_client
    domain_extractor.client = mock_client
    
    with pytest.raises(httpx.HTTPStatusError, match="401 Unauthorized"):
        domain_extractor._query_llm_with_retry("test prompt")
    assert mock_client.chat.completions.create.call_count == 3  # Should retry 3 times

@patch('faster.domain_knowledge.DomainKnowledgeExtractor._query_llm_with_retry')
def test_extract_knowledge_integration(mock_query, domain_extractor):
    """Test full knowledge extraction pipeline."""
    # Create larger sample data
    data = pd.DataFrame({
        'numeric_feature': range(20),
        'categorical_feature': ['A', 'B'] * 10,
        'text_feature': ['Sample text'] * 20,
        'target': [0, 1] * 10
    })
    
    mock_query.return_value = [
        {
            "feature_name": "numeric_feature",
            "importance": 0.8,
            "relationships": ["categorical_feature"],
            "suggested_transformations": ["log", "zscore"],
            "rationale": "High importance numeric feature"
        }
    ]
    
    insights = domain_extractor.extract_knowledge(
        data=data,
        target_column="target",
        problem_description="Test classification problem",
        domain_context="Test domain"
    )
    
    assert len(insights) == 1
    assert all(isinstance(insight, DomainInsight) for insight in insights)
    assert insights[0].feature_name == "numeric_feature"
    assert insights[0].importance == 0.8

def test_default_prompt_config(domain_extractor):
    """Test default prompt configuration."""
    config = domain_extractor._default_prompt_config()
    assert isinstance(config, PromptConfig)
    assert "{data_summary}" in config.context_template
    assert "{initial_insights}" in config.expert_template
    assert "{domain_context}" in config.feature_suggestion_template
    assert "{suggested_features}" in config.validation_template 