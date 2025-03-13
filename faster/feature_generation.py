"""Feature generation module for creating and transforming features."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from faster.utils.logging import get_logger
from faster.domain_knowledge import DomainInsight

logger = get_logger(__name__)

@dataclass
class TransformationMetadata:
    """Metadata about a feature transformation."""
    
    original_features: List[str]
    transformation_type: str
    parameters: Dict[str, Any]
    rationale: str

class FeatureGenerator:
    """Generates and transforms features based on domain knowledge."""
    
    def __init__(self):
        """Initialize the feature generator."""
        self.transformations: Dict[str, TransformationMetadata] = {}
        self._scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self._text_vectorizers: Dict[str, TfidfVectorizer] = {}
    
    def generate_features(
        self,
        data: pd.DataFrame,
        domain_insights: List[DomainInsight],
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate new features based on domain insights.
        
        Args:
            data: Input DataFrame
            domain_insights: List of domain insights about features
            target_column: Optional name of target variable to exclude from transformations
            
        Returns:
            DataFrame with generated features
        """
        logger.info("Starting feature generation process")
        result_df = data.copy()
        
        try:
            # Apply basic transformations
            result_df = self._apply_basic_transformations(result_df, domain_insights, target_column)
            
            # Generate interaction features
            result_df = self._generate_interaction_features(result_df, domain_insights)
            
            # Apply domain-specific transformations
            result_df = self._apply_domain_transformations(result_df, domain_insights)
            
            # Generate text features if text columns present
            result_df = self._generate_text_features(result_df, domain_insights)
            
            logger.info(f"Generated {len(result_df.columns) - len(data.columns)} new features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in feature generation: {str(e)}", exc_info=True)
            raise
    
    def _apply_basic_transformations(
        self,
        data: pd.DataFrame,
        insights: List[DomainInsight],
        target_column: Optional[str],
    ) -> pd.DataFrame:
        """Apply basic numerical transformations."""
        result_df = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if target_column:
            numeric_cols = numeric_cols.drop(target_column)
        
        for col in numeric_cols:
            # Log transformation for skewed features
            if self._should_apply_log_transform(data[col]):
                result_df[f"log_{col}"] = np.log1p(data[col])
                self.transformations[f"log_{col}"] = TransformationMetadata(
                    original_features=[col],
                    transformation_type="log",
                    parameters={},
                    rationale="High skewness detected",
                )
            
            # Standard scaling
            result_df[f"scaled_{col}"] = self._fit_transform_scaler(data[col], col)
            self.transformations[f"scaled_{col}"] = TransformationMetadata(
                original_features=[col],
                transformation_type="standard_scale",
                parameters={},
                rationale="Normalize feature distribution",
            )
        
        return result_df
    
    def _generate_interaction_features(
        self,
        data: pd.DataFrame,
        insights: List[DomainInsight],
    ) -> pd.DataFrame:
        """Generate interaction features between related variables."""
        result_df = data.copy()
        
        for insight in insights:
            # Handle single relationship with multiply transformation
            if "multiply" in insight.suggested_transformations and insight.relationships:
                for related in insight.relationships:
                    if insight.feature_name in data.columns and related in data.columns:
                        interaction_name = f"multiply_{insight.feature_name}_{related}"
                        result_df[interaction_name] = data[insight.feature_name] * data[related]
                        self.transformations[interaction_name] = TransformationMetadata(
                            original_features=[insight.feature_name, related],
                            transformation_type="multiply",
                            parameters={},
                            rationale=f"Multiplication suggested by domain knowledge",
                        )
            
            # Handle multiple relationships
            if len(insight.relationships) >= 2:
                for i, feat1 in enumerate(insight.relationships):
                    for feat2 in insight.relationships[i + 1:]:
                        if feat1 in data.columns and feat2 in data.columns:
                            interaction_name = f"interact_{feat1}_{feat2}"
                            result_df[interaction_name] = data[feat1] * data[feat2]
                            self.transformations[interaction_name] = TransformationMetadata(
                                original_features=[feat1, feat2],
                                transformation_type="interaction",
                                parameters={},
                                rationale=f"Interaction suggested by domain knowledge",
                            )
        
        return result_df
    
    def _apply_domain_transformations(
        self,
        data: pd.DataFrame,
        insights: List[DomainInsight],
    ) -> pd.DataFrame:
        """Apply transformations suggested by domain knowledge.
        
        Args:
            data: Input DataFrame
            insights: List of domain insights about features
            
        Returns:
            DataFrame with domain-specific transformations applied
        """
        result_df = data.copy()
        
        for insight in insights:
            feature = insight.feature_name
            if feature not in data.columns:
                logger.warning(f"Feature {feature} not found in dataset")
                continue
                
            for transform in insight.suggested_transformations:
                transform = transform.lower().strip()
                new_feature_name = None
                
                try:
                    if transform.startswith("log"):
                        # Log transformation
                        if data[feature].min() >= 0:
                            new_feature_name = f"log_{feature}"
                            result_df[new_feature_name] = np.log1p(data[feature])
                    
                    elif transform.startswith("sqrt"):
                        # Square root transformation
                        if data[feature].min() >= 0:
                            new_feature_name = f"sqrt_{feature}"
                            result_df[new_feature_name] = np.sqrt(data[feature])
                    
                    elif transform.startswith("square"):
                        # Square transformation
                        new_feature_name = f"square_{feature}"
                        result_df[new_feature_name] = np.square(data[feature])
                    
                    elif transform.startswith("cube"):
                        # Cube transformation
                        new_feature_name = f"cube_{feature}"
                        result_df[new_feature_name] = np.power(data[feature], 3)
                    
                    elif transform.startswith("bin"):
                        # Binning transformation
                        n_bins = 5  # Default number of bins
                        if "bins=" in transform:
                            try:
                                n_bins = int(transform.split("bins=")[1].split()[0])
                            except (IndexError, ValueError):
                                pass
                        
                        new_feature_name = f"binned_{feature}_{n_bins}"
                        result_df[new_feature_name] = pd.qcut(
                            data[feature],
                            q=n_bins,
                            labels=[f"bin_{i}" for i in range(n_bins)],
                            duplicates='drop'
                        )
                    
                    elif transform.startswith("zscore"):
                        # Z-score normalization
                        new_feature_name = f"zscore_{feature}"
                        result_df[new_feature_name] = (data[feature] - data[feature].mean()) / data[feature].std()
                    
                    elif transform.startswith("minmax"):
                        # Min-max scaling
                        new_feature_name = f"minmax_{feature}"
                        result_df[new_feature_name] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
                    
                    elif transform.startswith("power"):
                        # Power transformation
                        power = 2  # Default power
                        if "power=" in transform:
                            try:
                                power = float(transform.split("power=")[1].split()[0])
                            except (IndexError, ValueError):
                                pass
                        
                        new_feature_name = f"power_{power}_{feature}"
                        result_df[new_feature_name] = np.power(data[feature], power)
                    
                    elif transform.startswith("diff"):
                        # Difference with another feature
                        for related in insight.relationships:
                            if related in data.columns and related != feature:
                                new_feature_name = f"diff_{feature}_{related}"
                                result_df[new_feature_name] = data[feature] - data[related]
                    
                    elif transform.startswith("ratio"):
                        # Ratio with another feature
                        for related in insight.relationships:
                            if related in data.columns and related != feature:
                                new_feature_name = f"ratio_{feature}_{related}"
                                denominator = data[related]
                                # Avoid division by zero
                                if (denominator != 0).all():
                                    result_df[new_feature_name] = data[feature] / denominator
                    
                    # Record transformation metadata if successful
                    if new_feature_name:
                        self.transformations[new_feature_name] = TransformationMetadata(
                            original_features=[feature],
                            transformation_type=transform,
                            parameters={},
                            rationale=insight.rationale
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to apply transformation {transform} to {feature}: {str(e)}")
                    continue
        
        return result_df
    
    def _generate_text_features(
        self,
        data: pd.DataFrame,
        insights: List[DomainInsight],
    ) -> pd.DataFrame:
        """Generate features from text columns."""
        result_df = data.copy()
        text_cols = data.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if self._is_text_column(data[col]):
                vectorizer = TfidfVectorizer(max_features=10)
                text_features = vectorizer.fit_transform(data[col].fillna(''))
                feature_names = [f"tfidf_{col}_{i}" for i in range(text_features.shape[1])]
                
                for i, name in enumerate(feature_names):
                    result_df[name] = text_features[:, i].toarray()
                    self.transformations[name] = TransformationMetadata(
                        original_features=[col],
                        transformation_type="tfidf",
                        parameters={"index": i},
                        rationale="Text feature extraction",
                    )
                
                self._text_vectorizers[col] = vectorizer
        
        return result_df
    
    @staticmethod
    def _should_apply_log_transform(series: pd.Series) -> bool:
        """Check if log transform should be applied based on skewness."""
        return abs(series.skew()) > 1.0
    
    def _fit_transform_scaler(self, series: pd.Series, column_name: str) -> pd.Series:
        """Fit and transform using StandardScaler."""
        scaler = StandardScaler()
        self._scalers[column_name] = scaler
        return pd.Series(
            scaler.fit_transform(series.values.reshape(-1, 1)).flatten(),
            index=series.index,
        )
    
    @staticmethod
    def _is_text_column(series: pd.Series) -> bool:
        """Check if a column contains text data."""
        sample = series.dropna().head(100)
        return all(isinstance(x, str) and len(x.split()) > 3 for x in sample) 