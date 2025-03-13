"""Feature selection module for selecting optimal feature subsets."""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from faster.utils.logging import get_logger
from faster.statistical_evaluation import FeatureStatistics

logger = get_logger(__name__)

@dataclass
class SelectionCriteria:
    """Criteria for feature selection."""
    
    p_value_threshold: float = 0.05
    min_effect_size: float = 0.1
    max_correlation: float = 0.9
    min_mutual_info: float = 0.01
    max_features: Optional[int] = None

@dataclass
class SelectionResult:
    """Results of feature selection process."""
    
    selected_features: List[str]
    selection_scores: Dict[str, float]
    removed_features: Dict[str, str]  # feature -> reason for removal
    statistics: Dict[str, FeatureStatistics]

class FeatureSelector:
    """Selects optimal feature subset based on statistical and ML criteria."""
    
    def __init__(
        self,
        criteria: Optional[SelectionCriteria] = None,
        random_state: int = 42,
    ):
        """Initialize the feature selector.
        
        Args:
            criteria: Selection criteria configuration
            random_state: Random state for reproducibility
        """
        self.criteria = criteria or SelectionCriteria()
        self.random_state = random_state
    
    def select_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_stats: List[FeatureStatistics],
        is_classification: bool = True,
    ) -> SelectionResult:
        """Select optimal feature subset.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            feature_stats: Statistical metrics for features
            is_classification: Whether this is a classification task
            
        Returns:
            Selection results including selected features and metadata
        """
        logger.info("Starting feature selection process")
        
        try:
            # Initialize tracking
            candidates = set(data.columns) - {target_column}
            if not candidates:
                logger.warning("No features available for selection")
                return SelectionResult(
                    selected_features=[],
                    selection_scores={},
                    removed_features={},
                    statistics={},
                )
                
            removed = {}
            stats_dict = {stat.feature_name: stat for stat in feature_stats}
            
            # Remove features based on statistical criteria
            candidates = self._filter_by_statistics(candidates, stats_dict, removed)
            
            if not candidates:
                logger.warning("No features passed statistical criteria")
                return SelectionResult(
                    selected_features=[],
                    selection_scores={},
                    removed_features=removed,
                    statistics=stats_dict,
                )
            
            # Remove highly correlated features
            candidates = self._filter_correlations(data[list(candidates)], candidates, removed)
            
            if not candidates:
                logger.warning("No features passed correlation filtering")
                return SelectionResult(
                    selected_features=[],
                    selection_scores={},
                    removed_features=removed,
                    statistics=stats_dict,
                )
            
            # Apply ML-based selection
            final_features, importance_scores = self._ml_based_selection(
                data[list(candidates)],
                data[target_column],
                is_classification,
            )
            
            return SelectionResult(
                selected_features=final_features,
                selection_scores=importance_scores,
                removed_features=removed,
                statistics=stats_dict,
            )
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}", exc_info=True)
            raise
    
    def _filter_by_statistics(
        self,
        candidates: Set[str],
        stats: Dict[str, FeatureStatistics],
        removed: Dict[str, str],
    ) -> Set[str]:
        """Filter features based on statistical criteria."""
        result = candidates.copy()
        
        for feature in candidates:
            if feature not in stats:
                continue
                
            stat = stats[feature]
            
            # Check p-value
            if stat.p_value > self.criteria.p_value_threshold:
                removed[feature] = f"p-value ({stat.p_value:.4f}) > threshold ({self.criteria.p_value_threshold})"
                result.remove(feature)
                continue
            
            # Check effect size
            if stat.effect_size < self.criteria.min_effect_size:
                removed[feature] = f"effect size ({stat.effect_size:.4f}) < threshold ({self.criteria.min_effect_size})"
                result.remove(feature)
                continue
            
            # Check mutual information
            if stat.mutual_information < self.criteria.min_mutual_info:
                removed[feature] = f"mutual information ({stat.mutual_information:.4f}) < threshold ({self.criteria.min_mutual_info})"
                result.remove(feature)
        
        return result
    
    def _filter_correlations(
        self,
        data: pd.DataFrame,
        candidates: Set[str],
        removed: Dict[str, str],
    ) -> Set[str]:
        """Remove highly correlated features."""
        result = candidates.copy()
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Find highly correlated feature pairs
        for i in range(len(corr_matrix.columns)):
            if len(result) <= 1:
                break
                
            feat_i = corr_matrix.columns[i]
            if feat_i not in result:
                continue
                
            for j in range(i + 1, len(corr_matrix.columns)):
                feat_j = corr_matrix.columns[j]
                if feat_j not in result:
                    continue
                    
                correlation = corr_matrix.iloc[i, j]
                if correlation > self.criteria.max_correlation:
                    # Remove feature with lower variance
                    var_i = data[feat_i].var()
                    var_j = data[feat_j].var()
                    
                    if var_i >= var_j:
                        removed[feat_j] = f"high correlation with {feat_i} ({correlation:.4f})"
                        result.remove(feat_j)
                    else:
                        removed[feat_i] = f"high correlation with {feat_j} ({correlation:.4f})"
                        result.remove(feat_i)
                        break
        
        return result
    
    def _ml_based_selection(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        is_classification: bool,
    ) -> tuple[List[str], Dict[str, float]]:
        """Select features using ML-based importance scores."""
        if features.empty:
            logger.warning("No features available for ML-based selection")
            return [], {}
            
        # Initial filtering using univariate selection
        k = self.criteria.max_features or len(features.columns)
        selector = SelectKBest(
            score_func=f_classif if is_classification else f_regression,
            k=min(k, len(features.columns)),
        )
        selector.fit(features, target)
        
        # Get importance scores from Random Forest
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            
        model.fit(features, target)
        
        # Combine both scores
        importance_scores = {}
        for i, feature in enumerate(features.columns):
            # Normalize and combine both scores
            univariate_score = selector.scores_[i] / max(selector.scores_)
            rf_score = model.feature_importances_[i]
            importance_scores[feature] = (univariate_score + rf_score) / 2
        
        # Select top features
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        if self.criteria.max_features:
            selected = [f[0] for f in sorted_features[:self.criteria.max_features]]
        else:
            selected = [f[0] for f in sorted_features]
        
        return selected, importance_scores 