"""Statistical evaluation module for assessing feature significance."""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from statsmodels.stats.multitest import multipletests

from faster.utils.logging import get_logger
from faster.utils.validation import validate_dataframe

logger = get_logger(__name__)

@dataclass
class FeatureStatistics:
    """Statistical metrics for a feature."""
    
    feature_name: str
    correlation: Optional[float]
    p_value: float
    effect_size: float
    mutual_information: float
    test_method: str
    assumptions_met: Dict[str, bool]
    warnings: List[str]

class StatisticalEvaluator:
    """Evaluates statistical significance of features."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = "fdr_bh",
    ):
        """Initialize the statistical evaluator.
        
        Args:
            alpha: Significance level for statistical tests
            correction_method: Multiple testing correction method
        """
        self.alpha = alpha
        self.correction_method = correction_method
    
    def evaluate_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
    ) -> List[FeatureStatistics]:
        """Evaluate statistical significance of features.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            categorical_columns: List of categorical column names
            
        Returns:
            List of feature statistics
        """
        validate_dataframe(data, target_column)
        logger.info("Starting statistical evaluation of features")
        
        try:
            categorical_columns = categorical_columns or []
            results = []
            
            # Evaluate each feature
            for column in data.columns:
                if column != target_column:
                    stats = self._evaluate_single_feature(
                        data[column],
                        data[target_column],
                        column,
                        column in categorical_columns,
                    )
                    results.append(stats)
            
            # Apply multiple testing correction
            self._apply_multiple_testing_correction(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in statistical evaluation: {str(e)}", exc_info=True)
            raise
    
    def _evaluate_single_feature(
        self,
        feature: pd.Series,
        target: pd.Series,
        feature_name: str,
        is_categorical: bool,
    ) -> FeatureStatistics:
        """Evaluate statistical significance of a single feature."""
        assumptions = self._check_assumptions(feature, target, is_categorical)
        warnings = []
        
        # Choose appropriate test based on data characteristics
        if is_categorical:
            correlation, p_value, effect_size = self._categorical_test(feature, target)
            test_method = "chi_square"
        elif not assumptions["normality"]:
            correlation, p_value, effect_size = self._nonparametric_test(feature, target)
            test_method = "spearman"
            warnings.append("Non-normal distribution detected, using non-parametric test")
        else:
            correlation, p_value, effect_size = self._parametric_test(feature, target)
            test_method = "pearson"
        
        # Calculate mutual information
        mi_score = self._calculate_mutual_information(feature, target, is_categorical)
        
        return FeatureStatistics(
            feature_name=feature_name,
            correlation=correlation,
            p_value=p_value,
            effect_size=effect_size,
            mutual_information=mi_score,
            test_method=test_method,
            assumptions_met=assumptions,
            warnings=warnings,
        )
    
    def _check_assumptions(
        self,
        feature: pd.Series,
        target: pd.Series,
        is_categorical: bool,
    ) -> Dict[str, bool]:
        """Check statistical assumptions for the feature."""
        if is_categorical:
            return {
                "min_frequency": all(feature.value_counts() >= 5),
                "independence": True,  # Assumed, should be verified by study design
            }
        
        return {
            "normality": self._check_normality(feature),
            "linearity": self._check_linearity(feature, target),
            "homoscedasticity": self._check_homoscedasticity(feature, target),
        }
    
    def _parametric_test(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Tuple[float, float, float]:
        """Perform parametric statistical test (Pearson correlation)."""
        correlation, p_value = stats.pearsonr(feature, target)
        effect_size = abs(correlation)  # For Pearson's r, correlation is the effect size
        return correlation, p_value, effect_size
    
    def _nonparametric_test(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Tuple[float, float, float]:
        """Perform non-parametric statistical test (Spearman correlation)."""
        correlation, p_value = stats.spearmanr(feature, target)
        effect_size = abs(correlation)
        return correlation, p_value, effect_size
    
    def _categorical_test(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Tuple[float, float, float]:
        """Perform categorical statistical test (Chi-square test)."""
        contingency = pd.crosstab(feature, target)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        effect_size = np.sqrt(chi2 / (len(feature) * (min(contingency.shape) - 1)))  # Cramer's V
        return None, p_value, effect_size
    
    def _calculate_mutual_information(
        self,
        feature: pd.Series,
        target: pd.Series,
        is_categorical: bool,
    ) -> float:
        """Calculate mutual information score."""
        feature_reshaped = feature.values.reshape(-1, 1)
        if is_categorical:
            return mutual_info_classif(feature_reshaped, target, discrete_features=True)[0]
        return mutual_info_regression(feature_reshaped, target)[0]
    
    def _apply_multiple_testing_correction(self, results: List[FeatureStatistics]) -> None:
        """Apply multiple testing correction to p-values."""
        p_values = [r.p_value for r in results]
        rejected, p_corrected, _, _ = multipletests(
            p_values,
            alpha=self.alpha,
            method=self.correction_method,
        )
        
        for result, p_corrected in zip(results, p_corrected):
            result.p_value = p_corrected
    
    @staticmethod
    def _check_normality(series: pd.Series) -> bool:
        """Check if data is normally distributed using Shapiro-Wilk test."""
        if len(series) < 3:
            return False
        _, p_value = stats.shapiro(series)
        return p_value > 0.05
    
    @staticmethod
    def _check_linearity(feature: pd.Series, target: pd.Series) -> bool:
        """Check linearity assumption using Ramsey's RESET test."""
        # Simplified version - checks if quadratic term improves fit
        X = feature.values.reshape(-1, 1)
        X2 = np.square(X)
        linear_residuals = np.polyfit(X.flatten(), target, 1)[0]
        quadratic_residuals = np.polyfit(X.flatten(), target, 2)[0]
        return abs(linear_residuals - quadratic_residuals) < 0.1
    
    @staticmethod
    def _check_homoscedasticity(feature: pd.Series, target: pd.Series) -> bool:
        """Check homoscedasticity using Breusch-Pagan test."""
        # Simplified version - checks if residuals variance is constant
        coeffs = np.polyfit(feature, target, 1)
        residuals = target - np.polyval(coeffs, feature)
        _, p_value = stats.spearmanr(feature, np.square(residuals))
        return p_value > 0.05 