"""Feature selection module for selecting optimal feature subsets."""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.base import clone
from statsmodels.stats.outliers_influence import variance_inflation_factor

from faster.utils.logging import get_logger
from faster.statistical_evaluation import FeatureStatistics

logger = get_logger(__name__)

@dataclass
class SelectionCriteria:
    """Criteria for feature selection."""
    
    p_value_threshold: float = 0.05
    min_effect_size: float = 0.1
    max_correlation: float = 0.8  # Reduced from 0.9 to be more conservative
    min_mutual_info: float = 0.01
    max_features: Optional[int] = None
    min_importance_score: float = 0.02  # Minimum importance score to keep a feature
    vif_threshold: float = 10.0  # VIF threshold for multicollinearity
    cv_folds: int = 5  # Number of cross-validation folds
    stability_threshold: float = 0.7  # Frequency threshold for stability selection

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
            
            # Check for multicollinearity using VIF
            candidates = self._check_multicollinearity(data[list(candidates)], candidates, removed)
            
            if not candidates:
                logger.warning("No features passed multicollinearity check")
                return SelectionResult(
                    selected_features=[],
                    selection_scores={},
                    removed_features=removed,
                    statistics=stats_dict,
                )
            
            # Apply ML-based selection with cross-validation
            final_features, importance_scores = self._ml_based_selection_cv(
                data[list(candidates)],
                data[target_column],
                is_classification,
            )
            
            # Apply stability selection
            if len(final_features) > 1:
                stable_features = self._stability_selection(
                    data[final_features],
                    data[target_column],
                    is_classification
                )
                
                # If stability selection filtered out features, update final list
                if stable_features and len(stable_features) < len(final_features):
                    # Update removed features
                    for feature in final_features:
                        if feature not in stable_features:
                            removed[feature] = f"Failed stability selection (inconsistent importance)"
                    
                    # Update final features
                    final_features = stable_features
            
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
        """Remove highly correlated features with improved algorithm."""
        result = candidates.copy()
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Sort features by their average correlation with other features
        avg_corr = corr_matrix.mean().sort_values(ascending=False)
        
        # Iteratively remove highly correlated features
        for feature in avg_corr.index:
            if feature not in result:
                continue
                
            correlated_features = []
            
            for other_feature in result:
                if feature != other_feature and corr_matrix.loc[feature, other_feature] > self.criteria.max_correlation:
                    correlated_features.append(other_feature)
            
            if correlated_features:
                # Compare feature importance/usefulness using mutual information or effect size
                keep_feature = feature
                for corr_feature in correlated_features:
                    # If correlated feature has significantly higher variance, keep it instead
                    var_ratio = data[corr_feature].var() / data[feature].var()
                    if var_ratio > 1.5:  # Significantly higher variance
                        keep_feature = corr_feature
                        break
                
                # Remove all correlated features except the one to keep
                for f in correlated_features + [feature]:
                    if f != keep_feature and f in result:
                        removed[f] = f"high correlation with {keep_feature} ({corr_matrix.loc[f, keep_feature]:.4f})"
                        result.remove(f)
        
        return result
    
    def _check_multicollinearity(
        self, 
        data: pd.DataFrame,
        candidates: Set[str],
        removed: Dict[str, str]
    ) -> Set[str]:
        """Check for multicollinearity using Variance Inflation Factor (VIF)."""
        result = candidates.copy()
        
        try:
            # Only perform VIF check if we have enough features and samples
            if len(data.columns) < 2 or len(data) <= len(data.columns):
                return result
            
            # Calculate VIF for each feature
            features = list(result)
            X = data[features].copy()
            
            # Add constant for VIF calculation
            X_with_const = X.copy()
            X_with_const = sm_add_constant(X_with_const)
            
            # Calculate VIF iteratively and remove features with high VIF
            while True:
                if len(X.columns) < 2:
                    break
                    
                vifs = {}
                for i, col in enumerate(X.columns):
                    try:
                        vifs[col] = variance_inflation_factor(X_with_const.values, i + 1)  # +1 for constant
                    except:
                        vifs[col] = 0  # In case of errors (e.g., perfect collinearity)
                
                # Get maximum VIF
                max_vif_feature = max(vifs.items(), key=lambda x: x[1])
                
                # If max VIF exceeds threshold, remove feature
                if max_vif_feature[1] > self.criteria.vif_threshold:
                    feature_to_remove = max_vif_feature[0]
                    removed[feature_to_remove] = f"high VIF ({max_vif_feature[1]:.2f}) > threshold ({self.criteria.vif_threshold})"
                    result.remove(feature_to_remove)
                    
                    # Update X for next iteration
                    X = X.drop(columns=[feature_to_remove])
                    X_with_const = X.copy()
                    X_with_const = sm_add_constant(X_with_const)
                else:
                    break
                    
        except Exception as e:
            logger.warning(f"VIF calculation failed: {str(e)}. Skipping multicollinearity check.")
            
        return result
    
    def _ml_based_selection_cv(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        is_classification: bool,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using ML-based importance with cross-validation."""
        if features.empty:
            logger.warning("No features available for ML-based selection")
            return [], {}
        
        # Create cross-validation splitter
        cv = StratifiedKFold(n_splits=self.criteria.cv_folds, shuffle=True, random_state=self.random_state) if is_classification else KFold(n_splits=self.criteria.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize models for importance evaluation
        if is_classification:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            linear_model = LogisticRegression(random_state=self.random_state, penalty='l1', solver='liblinear', C=1.0)
        else:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            linear_model = Lasso(alpha=0.01, random_state=self.random_state)
        
        # Calculate feature importance across CV folds
        rf_importances = np.zeros(features.shape[1])
        linear_importances = np.zeros(features.shape[1])
        univariate_scores = np.zeros(features.shape[1])
        
        try:
            # Perform CV to calculate stable feature importances
            for train_idx, _ in cv.split(features, target):
                X_train, y_train = features.iloc[train_idx], target.iloc[train_idx]
                
                # RandomForest importance
                rf_clone = clone(rf_model)
                rf_clone.fit(X_train, y_train)
                rf_importances += rf_clone.feature_importances_
                
                # Linear model importance (coefficients)
                try:
                    linear_clone = clone(linear_model)
                    linear_clone.fit(X_train, y_train)
                    coefs = np.abs(linear_clone.coef_)
                    if coefs.ndim > 1:  # For multi-class classification
                        coefs = np.mean(coefs, axis=0)
                    linear_importances += coefs
                except:
                    pass  # Skip if linear model fails
                
                # Univariate statistical test
                score_func = f_classif if is_classification else f_regression
                f_stats, _ = score_func(X_train, y_train)
                univariate_scores += f_stats / np.max(f_stats)  # Normalize
        
        except Exception as e:
            logger.warning(f"Error in cross-validation importance calculation: {str(e)}")
            # Fall back to single model importance
            rf_model.fit(features, target)
            rf_importances = rf_model.feature_importances_ * self.criteria.cv_folds
        
        # Average importances across folds
        rf_importances /= self.criteria.cv_folds
        linear_importances /= self.criteria.cv_folds
        univariate_scores /= self.criteria.cv_folds
        
        # Combine different importance metrics
        combined_importances = np.zeros(features.shape[1])
        weights = [0.5, 0.3, 0.2]  # Weights for RF, linear, univariate
        
        for i in range(features.shape[1]):
            combined_importances[i] = (
                weights[0] * rf_importances[i] + 
                weights[1] * (linear_importances[i] if i < len(linear_importances) else 0) + 
                weights[2] * univariate_scores[i]
            )
        
        # Create importance dictionary
        importance_scores = {}
        for i, feature in enumerate(features.columns):
            importance_scores[feature] = combined_importances[i]
        
        # Apply minimum importance threshold
        sorted_features = sorted(
            [(feature, score) for feature, score in importance_scores.items() 
             if score > self.criteria.min_importance_score],
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Apply max_features limit if specified
        if self.criteria.max_features and len(sorted_features) > self.criteria.max_features:
            selected = [f[0] for f in sorted_features[:self.criteria.max_features]]
        else:
            selected = [f[0] for f in sorted_features]
        
        return selected, importance_scores
    
    def _stability_selection(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        is_classification: bool,
    ) -> List[str]:
        """Perform stability selection to identify consistently important features."""
        if len(features.columns) <= 1:
            return list(features.columns)
            
        n_features = len(features.columns)
        n_bootstraps = 10  # Number of bootstrap samples
        
        # Initialize selection frequency tracker
        selection_frequency = {feature: 0 for feature in features.columns}
        
        # Set up bootstrapping
        n_samples = int(0.8 * len(features))  # Use 80% of data in each bootstrap
        
        for i in range(n_bootstraps):
            # Create bootstrap sample
            indices = np.random.choice(len(features), size=n_samples, replace=True)
            X_boot = features.iloc[indices]
            y_boot = target.iloc[indices]
            
            # Build model for feature selection
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=self.random_state + i)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=self.random_state + i)
                
            # Fit model
            model.fit(X_boot, y_boot)
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Get top 75% of features
            n_to_select = max(1, int(0.75 * n_features))
            top_indices = np.argsort(importances)[-n_to_select:]
            
            # Update feature selection frequency
            for idx in top_indices:
                feature = features.columns[idx]
                selection_frequency[feature] += 1
        
        # Calculate selection frequency
        for feature in selection_frequency:
            selection_frequency[feature] /= n_bootstraps
        
        # Select features that appear in at least threshold% of bootstrap samples
        selected_features = [
            feature for feature, freq in selection_frequency.items() 
            if freq >= self.criteria.stability_threshold
        ]
        
        # If no features meet threshold, take top 3 most frequently selected
        if not selected_features and features.columns.any():
            sorted_features = sorted(
                selection_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected_features = [f[0] for f in sorted_features[:min(3, len(sorted_features))]]
        
        return selected_features

# Helper function to add constant for VIF calculation
def sm_add_constant(data):
    """Add constant column for statsmodels functions without importing statsmodels."""
    const = pd.Series(1, index=data.index, name="const")
    return pd.concat([const, data], axis=1) 