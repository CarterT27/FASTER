"""Domain knowledge extraction module using LLMs."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging
import uuid
import json
import os
import time

import pandas as pd
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from pydantic import BaseModel

from faster.utils.logging import get_logger
from faster.utils.validation import validate_dataframe

logger = get_logger(__name__)

class DomainInsight(BaseModel):
    """Structure for storing domain-specific insights about features."""
    
    feature_name: str
    importance: float
    relationships: List[str]
    suggested_transformations: List[str]
    rationale: str

@dataclass
class PromptConfig:
    """Configuration for LLM prompting."""
    
    context_template: str
    expert_template: str
    feature_suggestion_template: str
    validation_template: str

class DomainKnowledgeExtractor:
    """Extracts domain knowledge from data using LLMs."""

    def __init__(
        self,
        model_name: str = "deepseek/deepseek-chat:free",
        temperature: float = 0.0,
        prompt_config: Optional[PromptConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the domain knowledge extractor.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM sampling
            prompt_config: Custom prompt configuration
            api_key: OpenRouter API key (overrides environment variable if provided)
        """
        # Use provided API key or get from environment
        if api_key:
            self.api_key = api_key.strip()
        elif "OPENROUTER_API_KEY" in os.environ:
            self.api_key = os.environ["OPENROUTER_API_KEY"].strip()
        else:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required or api_key must be provided. "
                "Get your API key from https://openrouter.ai/keys"
            )
            
        if not self.api_key:
            raise ValueError("API key is empty")

        self.model_name = model_name
        self.temperature = temperature
        
        # Configure OpenRouter client with API key
        # Set up the client according to OpenRouter documentation
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        
        # Log key information (masked)
        masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "***"
        logger.debug(f"Initialized OpenRouter client with key: {masked_key} (length: {len(self.api_key)})")
        
        self.prompt_config = prompt_config or self._default_prompt_config()
        self._conversation_history: List[BaseMessage] = []
    
    def extract_knowledge(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_description: str,
        request_id: Optional[str] = None,
        domain_context: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> List[DomainInsight]:
        """
        Extract domain knowledge and feature insights.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            problem_description (str): Description of the problem
            request_id (str, optional): Unique ID to track this request
            domain_context (str, optional): Additional domain context
            categorical_columns (List[str], optional): List of categorical columns
            
        Returns:
            List[DomainInsight]: List of domain insights
        """
        # Generate a request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        logger.info(f"Starting domain knowledge extraction with request ID: {request_id}")
        
        # Ensure data quality
        if data.empty:
            logger.warning("Empty dataset provided, returning empty insights")
            return []
            
        # Handle missing target column
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            return []
            
        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = []
            for col in data.columns:
                if col == target_column:
                    continue
                    
                if pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                    categorical_columns.append(col)
                # Also detect low-cardinality numeric features as potential categorical features
                elif pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() < 10:
                    categorical_columns.append(col)
        
        # Generate data summary for LLM context
        try:
            data_summary = self._generate_data_summary(
                data, 
                target_column
            )
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            # Create minimal data summary
            data_summary = {
                "n_rows": len(data),
                "n_columns": len(data.columns),
                "features": list(data.columns),
                "data_types": {col: str(data[col].dtype) for col in data.columns},
                "categorical_columns": categorical_columns,
                "target": target_column
            }
        
        # Create prompt for LLM
        # Add domain context to problem description if provided
        full_context = problem_description
        if domain_context:
            full_context += f"\n\nAdditional context: {domain_context}"
            
        prompt = self.prompt_config.context_template.format(
            data_summary=json.dumps(data_summary, indent=2),
            problem_description=full_context,
        )
        
        # Query LLM for insights
        try:
            logger.info(f"Querying LLM (attempt 1/3)")
            response = self._query_llm_with_retry(prompt)
            
            # Convert raw insights to DomainInsight objects
            domain_insights = []
            for raw_insight in response:
                try:
                    # Validate and clean the raw insight
                    if not isinstance(raw_insight, dict):
                        logger.warning(f"Invalid insight format: {raw_insight}")
                        continue
                        
                    # Ensure all required fields are present
                    required_fields = ["feature_name", "importance", "relationships", 
                                      "suggested_transformations", "rationale"]
                    if not all(field in raw_insight for field in required_fields):
                        logger.warning(f"Missing required fields in insight: {raw_insight}")
                        continue
                        
                    # Ensure feature actually exists in the dataset
                    if raw_insight["feature_name"] not in data.columns:
                        logger.warning(f"Feature {raw_insight['feature_name']} not found in dataset")
                        continue
                        
                    # Create DomainInsight object
                    insight = DomainInsight(
                        feature_name=raw_insight["feature_name"],
                        importance=float(raw_insight["importance"]),
                        relationships=raw_insight["relationships"],
                        suggested_transformations=raw_insight["suggested_transformations"],
                        rationale=raw_insight["rationale"]
                    )
                    domain_insights.append(insight)
                except Exception as insight_error:
                    logger.warning(f"Error processing insight: {str(insight_error)}")
                    continue
                    
            return domain_insights
            
        except Exception as e:
            logger.error(f"Error extracting domain knowledge: {str(e)}")
            # Return empty list on error
            return []
    
    def _generate_data_summary(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Generate a summary of the dataset for LLM context."""
        return {
            "n_samples": len(data),
            "n_features": len(data.columns) - 1,
            "feature_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "missing_values": data.isnull().sum().to_dict(),
            "target_distribution": data[target_column].describe().to_dict(),
        }
    
    def _query_llm_with_retry(self, prompt: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Query LLM with retry logic."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Querying LLM (attempt {attempt+1}/{max_retries})")
                
                # Log API configuration (with masked key)
                api_key = self.client.api_key
                # Handle both string keys and mock objects
                if isinstance(api_key, str):
                    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
                    logger.debug(f"Using API key: {masked_key} (length: {len(api_key)})")
                else:
                    logger.debug(f"Using API key: [MOCK OBJECT]")
                
                logger.debug(f"Base URL: {self.client.base_url}")
                logger.debug(f"Model: {self.model_name}")
                
                # Make the API call with the app name headers
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/CarterT27/FASTER",
                        "X-Title": "FASTER Feature Selection Tool"
                    }
                )
                
                logger.info(f"LLM query successful")
                return self._parse_llm_response(response.choices[0].message.content)
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"LLM query attempt {attempt + 1} failed: {error_msg}")
                
                # Add more detailed error information
                if hasattr(e, 'response'):
                    status_code = getattr(e.response, 'status_code', 'unknown')
                    logger.warning(f"Status code: {status_code}")
                    
                    # Try to extract response body
                    try:
                        response_text = e.response.text
                        logger.warning(f"Response body: {response_text}")
                    except:
                        pass
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} LLM query attempts failed")
                    raise
                
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def _refine_insights(
        self,
        initial_insights: List[Dict[str, Any]],
        data: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Refine initial insights through expert prompting."""
        # Implementation details for insight refinement
        return initial_insights
    
    @staticmethod
    def _parse_llm_response(response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured insights.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            List of dictionaries containing structured insights
            
        Raises:
            ValueError: If response cannot be parsed into valid insights
        """
        try:
            # First try to parse as direct JSON
            try:
                insights = json.loads(response)
                if isinstance(insights, list):
                    # Validate required fields
                    required_fields = ["feature_name", "importance", "relationships", 
                                     "suggested_transformations", "rationale"]
                    for insight in insights:
                        if not all(field in insight for field in required_fields):
                            raise ValueError("Missing required fields")
                    return insights
            except json.JSONDecodeError:
                pass
            
            # If not direct JSON, try to extract JSON-like structure from text
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.finditer(json_pattern, response)
            insights = []
            
            for match in matches:
                try:
                    insight = json.loads(match.group())
                    if all(key in insight for key in ["feature_name", "importance", "relationships", 
                                                    "suggested_transformations", "rationale"]):
                        insights.append(insight)
                except json.JSONDecodeError:
                    continue
            
            if not insights:
                raise ValueError("No valid insights found in LLM response")
            
            # Validate and clean insights
            cleaned_insights = []
            for insight in insights:
                cleaned_insight = {
                    "feature_name": str(insight["feature_name"]),
                    "importance": float(insight["importance"]),
                    "relationships": [str(r) for r in insight["relationships"]],
                    "suggested_transformations": [str(t) for t in insight["suggested_transformations"]],
                    "rationale": str(insight["rationale"])
                }
                cleaned_insights.append(cleaned_insight)
            
            return cleaned_insights
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise ValueError(f"Could not parse LLM response: {str(e)}")
    
    @staticmethod
    def _default_prompt_config() -> PromptConfig:
        """Default prompting configuration."""
        context_template = """
        You are an expert data scientist analyzing a dataset with the following characteristics:
        {data_summary}
        
        The machine learning problem is:
        {problem_description}
        
        Please analyze each feature and provide insights in the following JSON format:
        [
            {{
                "feature_name": "name_of_feature",
                "importance": 0.0-1.0,
                "relationships": ["related_feature1", "related_feature2"],
                "suggested_transformations": ["transformation1", "transformation2"],
                "rationale": "Detailed explanation of why these transformations would be useful"
            }}
        ]
        
        Focus on:
        1. Identifying potential feature interactions
        2. Suggesting appropriate transformations (log, polynomial, binning, etc.)
        3. Explaining the domain relevance of each feature
        4. Noting any data quality concerns
        
        Provide concrete, actionable insights that can be used for feature engineering.
        """
        
        expert_template = """
        Review and enhance the following feature insights:
        {initial_insights}
        
        Consider:
        1. Are the suggested transformations appropriate for the data types?
        2. Are there any missing important feature interactions?
        3. Are the importance scores well-justified?
        4. Could any domain-specific transformations improve the features?
        
        Provide refined insights in the same JSON format.
        """
        
        feature_suggestion_template = """
        Based on the domain knowledge and data characteristics:
        {domain_context}
        {data_summary}
        
        Suggest additional derived features that could be valuable, considering:
        1. Common domain-specific indicators or ratios
        2. Temporal patterns or seasonality
        3. Categorical feature combinations
        4. Text-derived features
        
        Provide suggestions in the standard JSON format.
        """
        
        validation_template = """
        Validate the following feature engineering suggestions:
        {suggested_features}
        
        Check for:
        1. Statistical validity of transformations
        2. Potential data leakage
        3. Computational feasibility
        4. Business logic consistency
        
        Return a filtered list of valid suggestions in the standard JSON format.
        """
        
        return PromptConfig(
            context_template=context_template,
            expert_template=expert_template,
            feature_suggestion_template=feature_suggestion_template,
            validation_template=validation_template,
        ) 