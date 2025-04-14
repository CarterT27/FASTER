"""Domain knowledge extraction module using LLMs."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging
import uuid
import json
import os
import time
import traceback

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

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        masked_key = (
            self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "***"
        )
        logger.debug(
            f"Initialized OpenRouter client with key: {masked_key} (length: {len(self.api_key)})"
        )

        self.prompt_config = prompt_config or self._default_prompt_config()
        self._conversation_history: List[BaseMessage] = []

    def extract_knowledge(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_description: str,
        request_id: Optional[str] = None,
    ) -> List[DomainInsight]:
        """Extract domain knowledge from LLM.

        Args:
            data: Input DataFrame
            target_column: Name of target variable
            problem_description: Description of the problem
            request_id: Unique ID for the request

        Returns:
            List of domain insights
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        logger.info(f"Extracting domain knowledge (request_id: {request_id})")

        prompt = self._generate_domain_prompt(data, target_column, problem_description)

        try:
            logger.info("Querying LLM for domain knowledge insights")
            response = self._query_llm_with_retry(prompt, request_id)

            insights = self._parse_insights(response)

            logger.info(f"Extracted {len(insights)} domain insights")
            return insights

        except Exception as e:
            logger.error(f"Error extracting domain knowledge: {str(e)}")
            logger.error(traceback.format_exc())

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

    def _query_llm_with_retry(
        self,
        prompt: str,
        request_id: str,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Query LLM with retry logic."""
        retries = 0
        backoff = initial_backoff

        while retries < max_retries:
            try:
                logger.info(
                    f"Querying LLM (attempt {retries + 1}/{max_retries}, request_id: {request_id})"
                )

                messages = [{"role": "user", "content": prompt}]

                client = self._get_client()

                start_time = time.time()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=0.95,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                end_time = time.time()

                logger.info(f"LLM query duration: {end_time - start_time:.2f}s")

                response_content = response.choices[0].message.content

                try:
                    import re

                    pattern = r"\[\s*{.*}\s*\]"
                    matches = re.search(pattern, response_content, re.DOTALL)

                    if matches:
                        json_str = matches.group(0)
                        parsed_response = json.loads(json_str)
                    else:
                        parsed_response = json.loads(response_content)

                        if isinstance(parsed_response, dict):
                            for key, value in parsed_response.items():
                                if (
                                    isinstance(value, list)
                                    and len(value) > 0
                                    and isinstance(value[0], dict)
                                ):
                                    parsed_response = value
                                    break

                    if not isinstance(parsed_response, list):
                        raise ValueError(f"Expected a list response, got {type(parsed_response)}")

                    return parsed_response

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {str(e)}")
                    logger.error(f"Response content: {response_content}")
                    raise ValueError(f"Failed to parse JSON response: {str(e)}")

            except Exception as e:
                retries += 1
                logger.warning(f"LLM query failed (attempt {retries}/{max_retries}): {str(e)}")

                if retries >= max_retries:
                    logger.error(f"Maximum retries reached. Query failed: {str(e)}")
                    raise

                sleep_time = backoff * (2 ** (retries - 1))
                logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

    def _refine_insights(
        self,
        initial_insights: List[Dict[str, Any]],
        data: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Refine initial insights through expert prompting."""

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
            try:
                insights = json.loads(response)
                if isinstance(insights, list):
                    required_fields = [
                        "feature_name",
                        "importance",
                        "relationships",
                        "suggested_transformations",
                        "rationale",
                    ]
                    for insight in insights:
                        if not all(field in insight for field in required_fields):
                            raise ValueError("Missing required fields")
                    return insights
            except json.JSONDecodeError:
                pass

            import re

            json_pattern = r"\{[^{}]*\}"
            matches = re.finditer(json_pattern, response)
            insights = []

            for match in matches:
                try:
                    insight = json.loads(match.group())
                    if all(
                        key in insight
                        for key in [
                            "feature_name",
                            "importance",
                            "relationships",
                            "suggested_transformations",
                            "rationale",
                        ]
                    ):
                        insights.append(insight)
                except json.JSONDecodeError:
                    continue

            if not insights:
                raise ValueError("No valid insights found in LLM response")

            cleaned_insights = []
            for insight in insights:
                cleaned_insight = {
                    "feature_name": str(insight["feature_name"]),
                    "importance": float(insight["importance"]),
                    "relationships": [str(r) for r in insight["relationships"]],
                    "suggested_transformations": [
                        str(t) for t in insight["suggested_transformations"]
                    ],
                    "rationale": str(insight["rationale"]),
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

    def _generate_domain_prompt(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_description: str,
    ) -> str:
        """Generate prompt for domain knowledge extraction."""

        sample_rows = min(5, len(data))
        data_sample = data.head(sample_rows).to_string()

        column_info = []
        for col in data.columns:
            dtype = data[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                stats = {
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "mean": data[col].mean(),
                    "median": data[col].median(),
                    "std": data[col].std(),
                    "missing": data[col].isna().sum(),
                }
                col_type = "numeric"
            else:
                stats = {
                    "unique_values": data[col].nunique(),
                    "top_values": list(data[col].value_counts().head(3).index),
                    "missing": data[col].isna().sum(),
                }
                col_type = "categorical"

            column_info.append(
                {
                    "name": col,
                    "type": col_type,
                    "stats": stats,
                    "is_target": col == target_column,
                }
            )

        column_descriptions = "\n".join(
            [
                f"- {info['name']}: {info['type']} column, "
                + (f"stats: {info['stats']}" if not info["is_target"] else "(target column)")
                for info in column_info
            ]
        )

        is_iris_dataset = (
            all(col in data.columns for col in ["sepal_width", "petal_length", "petal_width"])
            or "species" in data.columns
        )
        is_titanic_dataset = (
            all(col in data.columns for col in ["Pclass", "Sex", "Age", "Survived"])
            or "Fare" in data.columns
        )

        additional_context = ""
        if is_iris_dataset:
            additional_context = """
This appears to be the Iris dataset, so consider these important domain insights:
1. Petal dimensions (length and width) are strongly correlated with species
2. Setosa species has the smallest petals but relatively wide sepals
3. Virginica species has the largest petals and sepals
4. Petal dimensions tend to be more discriminative than sepal dimensions for species classification
5. Log transformations and ratios between petal and sepal dimensions may be valuable features
"""
        elif is_titanic_dataset:
            additional_context = """
This appears to be the Titanic dataset, so consider these important domain insights:
1. Gender was a primary factor in survival due to "women and children first" policy
2. Passenger class (Pclass) affected survival rates due to cabin location and preferential access to lifeboats
3. Age affected survival chances with children having priority
4. Family structure (SibSp, Parch) influenced survival, with small to medium sized families having better chances
5. Interactions between Age, Sex, and Pclass are particularly informative
6. Fare is highly correlated with Pclass and can be logarithmically transformed to better reflect its relationship with survival
"""

        return f"""You are a domain expert helping analyze a dataset for machine learning. 
Your task is to provide domain knowledge that can guide feature engineering.

{data_sample}

{column_descriptions}

{problem_description}

{additional_context}

Based on the data and problem description, generate domain knowledge insights for feature engineering. 
For each feature, include:
1. Its importance for predicting the target
2. Relationships with other features
3. Recommended transformations (e.g., log, one-hot, interactions, binning)
4. Rationale for why these transformations would be useful

Format your response as a list of JSON objects, one for each feature:
[
  {{
    "feature_name": <name>,
    "importance": <float 0-1>,
    "relationships": [<related_feature1>, <related_feature2>, ...],
    "suggested_transformations": [<transformation1>, <transformation2>, ...],
    "rationale": <explanation>
  }},
  ...
]

Include only valid transformations: log, zscore, min_max, binning, polynomial, one_hot, label, interaction.
Focus on the most important features first and provide at least 4-5 insights.
"""

    def _parse_insights(self, response: List[Dict[str, Any]]) -> List[DomainInsight]:
        """Parse and validate LLM insights into domain insight objects.

        Args:
            response: Raw LLM response

        Returns:
            List of validated domain insights
        """
        insights = []

        for raw_insight in response:
            try:
                if not isinstance(raw_insight, dict):
                    logger.warning(f"Invalid insight format: {raw_insight}")
                    continue

                required_fields = [
                    "feature_name",
                    "importance",
                    "relationships",
                    "suggested_transformations",
                    "rationale",
                ]
                if not all(field in raw_insight for field in required_fields):
                    logger.warning(f"Missing required fields in insight: {raw_insight}")
                    continue

                insight = DomainInsight(
                    feature_name=raw_insight["feature_name"],
                    importance=float(raw_insight["importance"]),
                    relationships=raw_insight["relationships"],
                    suggested_transformations=raw_insight["suggested_transformations"],
                    rationale=raw_insight["rationale"],
                )
                insights.append(insight)
            except Exception as insight_error:
                logger.warning(f"Error processing insight: {str(insight_error)}")
                continue

        return insights

    def _get_client(self):
        """Get or create an OpenAI client for API calls."""
        import openai

        if hasattr(self, "_openai_client") and self._openai_client is not None:
            return self._openai_client

        try:
            self._openai_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/CarterT27/FASTER",
                    "X-Title": "FASTER Feature Selection Tool",
                },
            )
            return self._openai_client
        except Exception as e:
            logger.error(f"Error creating OpenAI client: {str(e)}")
            raise

    def extract_domain_knowledge(
        self,
        problem_description: str,
        data_sample: pd.DataFrame,
        column_names: List[str],
        categorical_columns: Optional[List[str]] = None,
        domain_context: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> List[DomainInsight]:
        """Extract domain knowledge from LLM.

        Args:
            problem_description: Description of the problem
            data_sample: Sample of the dataset
            column_names: List of column names
            categorical_columns: Columns to treat as categorical
            domain_context: Additional domain context
            request_id: Unique ID for the request

        Returns:
            List of domain insights
        """

        target_column = column_names[-1] if column_names else None

        enhanced_problem_description = problem_description
        if domain_context:
            enhanced_problem_description = (
                f"{problem_description}\n\nAdditional domain context: {domain_context}"
            )

        return self.extract_knowledge(
            data=data_sample,
            target_column=target_column,
            problem_description=enhanced_problem_description,
            request_id=request_id,
        )
