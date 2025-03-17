"""Statistical evaluation module for assessing feature significance."""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

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
    predictive_power: Optional[float] = None  # Added measure of univariate predictive power

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

            if not categorical_columns:
                categorical_columns = self._detect_categorical_columns(data)
                logger.info(f"Auto-detected categorical columns: {categorical_columns}")

            is_classification = self._is_classification_problem(data[target_column])
            logger.info(f"Problem type: {'Classification' if is_classification else 'Regression'}")
            
            results = []

            for column in data.columns:
                if column != target_column:
                    stats = self._evaluate_single_feature(
                        column,
                        data[column],
                        data[target_column],
                    )
                    results.append(stats)

            self._apply_multiple_testing_correction(results)

            self._evaluate_predictive_power(data, target_column, results, is_classification)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in statistical evaluation: {str(e)}", exc_info=True)
            raise
    
    def _evaluate_single_feature(
        self,
        feature_name: str,
        feature_data: pd.Series,
        target: pd.Series,
    ) -> FeatureStatistics:
        """Evaluate statistical significance of a single feature."""

        if isinstance(feature_data, pd.DataFrame):

            if feature_data.shape[1] == 1:
                feature_data = feature_data.iloc[:, 0]
            else:


                self.logger.warning(f"Feature {feature_name} is a multi-column DataFrame. Using first column for evaluation.")
                feature_data = feature_data.iloc[:, 0]

        is_categorical = feature_data.dtype == 'object' or feature_data.dtype.name == 'category'
        is_categorical = is_categorical or (
            isinstance(feature_data, pd.Series) and 
            feature_data.str.contains('bin_').any() if hasattr(feature_data, 'str') else False
        )
        
        assumptions = self._check_assumptions(feature_data, target, is_categorical)
        warnings = []

        if is_categorical and self._is_classification_problem(target):
            correlation, p_value, effect_size = self._categorical_vs_categorical(feature_data, target)
            test_method = "chi_square"
        elif is_categorical and not self._is_classification_problem(target):
            correlation, p_value, effect_size = self._categorical_vs_continuous(feature_data, target)
            test_method = "anova"
        elif not is_categorical and self._is_classification_problem(target):
            correlation, p_value, effect_size = self._continuous_vs_categorical(feature_data, target)
            test_method = "point_biserial"
        else:  # Both continuous
            if not assumptions["normality"]:
                correlation, p_value, effect_size = self._nonparametric_test(feature_data, target)
                test_method = "spearman"
                warnings.append("Non-normal distribution detected, using non-parametric test")
            else:
                correlation, p_value, effect_size = self._parametric_test(feature_data, target)
                test_method = "pearson"

        mi_score = self._calculate_mutual_information(feature_data, target, is_categorical)

        if effect_size < 0.1 and p_value <= self.alpha:
            warnings.append(f"Statistically significant but small effect size ({effect_size:.3f})")

        if p_value > self.alpha and mi_score > 0.2:
            warnings.append(f"High mutual information but non-significant p-value, may indicate non-linear relationship")
        
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
    
    def _categorical_vs_categorical(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Tuple[float, float, float]:
        """Perform categorical statistical test (Chi-square test)."""
        contingency = pd.crosstab(feature, target)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        n = contingency.sum().sum()
        effect_size = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))  # Cramer's V
        return None, p_value, effect_size
    
    def _categorical_vs_continuous(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Tuple[float, float, float]:
        """
        Analyze relationship between categorical feature and continuous target.
        
        Args:
            feature: Categorical feature
            target: Continuous target variable
            
        Returns:
            tuple: (correlation, p_value, effect_size)
        """

        if isinstance(feature, pd.DataFrame):
            if feature.shape[1] == 1:
                feature = feature.iloc[:, 0]
            else:
                self.logger.warning(f"Multi-column categorical feature provided. Using first column.")
                feature = feature.iloc[:, 0]
        
        try:

            if pd.api.types.is_categorical_dtype(feature) or pd.api.types.is_object_dtype(feature):



                if hasattr(feature, 'str') and isinstance(feature.iloc[0], str) and feature.str.contains('bin_').any():
                    try:

                        numeric_feature = pd.Series(
                            [int(val.split('_')[1]) if isinstance(val, str) and 'bin_' in val 
                             else float('nan') for val in feature],
                            index=feature.index
                        )

                        return self._run_anova(numeric_feature, target)
                    except (IndexError, ValueError, AttributeError) as e:
                        self.logger.warning(f"Could not convert binned feature to numeric: {str(e)}")

                        pass


                try:
                    numeric_feature = pd.Categorical(feature).codes

                    if numeric_feature.max() < 0:
                        return 0.0, 1.0, 0.0
                        
                    return self._run_anova(numeric_feature, target)
                except Exception as e:
                    self.logger.warning(f"Error converting categorical to codes: {str(e)}")

                    pass

            groups = []
            categories = []

            for category in feature.unique():
                if pd.isna(category):
                    continue
                category_target = target[feature == category].dropna()
                if len(category_target) > 0:
                    groups.append(category_target)
                    categories.append(category)

            if len(groups) < 2:
                return 0.0, 1.0, 0.0

            try:
                f_statistic, p_value = stats.f_oneway(*groups)

                grand_mean = target.mean()
                ss_total = ((target - grand_mean) ** 2).sum()
                
                ss_between = sum(len(group) * ((group.mean() - grand_mean) ** 2) 
                                for group in groups)
                
                eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

                correlation = np.sqrt(eta_squared) if not np.isnan(eta_squared) else 0.0
                
                return correlation, p_value, eta_squared
            except Exception as e:
                self.logger.warning(f"ANOVA failed: {str(e)}")
                return 0.0, 1.0, 0.0
                
        except Exception as e:
            self.logger.error(f"Error in categorical vs continuous analysis: {str(e)}")
            return 0.0, 1.0, 0.0
            
    def _run_anova(self, numeric_feature: pd.Series, target: pd.Series) -> Tuple[float, float, float]:
        """Helper method to run ANOVA on numeric representation of features."""
        try:

            groups = []
            for value in sorted(numeric_feature.unique()):
                if pd.isna(value):
                    continue
                value_target = target[numeric_feature == value].dropna()
                if len(value_target) > 0:
                    groups.append(value_target)

            if len(groups) < 2:
                return 0.0, 1.0, 0.0

            f_statistic, p_value = stats.f_oneway(*groups)

            grand_mean = target.mean()
            ss_total = ((target - grand_mean) ** 2).sum()
            
            ss_between = sum(len(group) * ((group.mean() - grand_mean) ** 2) 
                            for group in groups)
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

            correlation = np.sqrt(eta_squared) if not np.isnan(eta_squared) else 0.0
            
            return correlation, p_value, eta_squared
        except Exception as e:
            self.logger.warning(f"ANOVA on numeric feature failed: {str(e)}")
            return 0.0, 1.0, 0.0
    
    def _continuous_vs_categorical(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Tuple[float, float, float]:
        """Point-biserial correlation for continuous feature vs categorical target."""

        target_numeric = pd.factorize(target)[0]

        correlation, p_value = stats.pointbiserialr(feature, target_numeric)
        effect_size = abs(correlation)
        
        return correlation, p_value, effect_size
    
    def _calculate_mutual_information(
        self,
        feature: pd.Series,
        target: pd.Series,
        is_categorical: bool,
    ) -> float:
        """Calculate mutual information score."""
        feature_reshaped = feature.values.reshape(-1, 1)
        is_target_categorical = self._is_classification_problem(target)
        
        if is_target_categorical:
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression
            
        return mi_func(feature_reshaped, target, discrete_features=[is_categorical])[0]
    
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
    
    def _evaluate_predictive_power(
        self,
        data: pd.DataFrame,
        target_column: str,
        results: List[FeatureStatistics],
        is_classification: bool,
    ) -> None:
        """Evaluate univariate predictive power of each feature using cross-validation."""
        target = data[target_column]

        cv = StratifiedKFold(n_splits=5) if is_classification else KFold(n_splits=5)
        
        for stat in results:
            feature = stat.feature_name
            X = data[[feature]].copy()

            if X.isnull().sum().sum() / len(X) > 0.2:
                stat.predictive_power = 0.0
                continue

            try:

                if is_classification:
                    base_score = data[target_column].value_counts(normalize=True).max()  # Majority class frequency
                    if self._is_categorical_feature(data[feature]):
                        model = DecisionTreeClassifier(max_depth=3)
                    else:
                        model = LogisticRegression()

                    if len(target.unique()) == 2:
                        scoring = 'roc_auc'
                    else:
                        scoring = 'accuracy'
                else:
                    base_score = 0.0  # R² of predicting the mean is 0
                    if self._is_categorical_feature(data[feature]):
                        model = DecisionTreeRegressor(max_depth=3)
                    else:
                        model = LinearRegression()
                    scoring = 'r2'

                cv_scores = cross_val_score(model, X, target, cv=cv, scoring=scoring)
                avg_score = np.mean(cv_scores)

                if is_classification and scoring == 'accuracy':

                    normalized_score = (avg_score - base_score) / (1 - base_score) if avg_score > base_score else 0
                elif is_classification and scoring == 'roc_auc':

                    normalized_score = (avg_score - 0.5) / 0.5 if avg_score > 0.5 else 0
                else:

                    normalized_score = max(0, avg_score)

                stat.predictive_power = min(1.0, normalized_score)
                
            except Exception as e:
                logger.warning(f"Error calculating predictive power for {feature}: {str(e)}")
                stat.predictive_power = 0.0
    
    @staticmethod
    def _check_normality(series: pd.Series) -> bool:
        """Check if data is normally distributed using Shapiro-Wilk test."""

        sample = series.dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000)
            
        if len(sample) < 3:
            return False
            
        try:
            _, p_value = stats.shapiro(sample)
            return p_value > 0.05
        except Exception:

            skew = abs(sample.skew())
            kurtosis = abs(sample.kurtosis())
            return skew < 1.0 and kurtosis < 3.0
    
    @staticmethod
    def _check_linearity(feature: pd.Series, target: pd.Series) -> bool:
        """Check linearity assumption using augmented Ramsey RESET test."""
        try:

            X = feature.values.reshape(-1, 1)
            X2 = np.power(X, 2)
            X3 = np.power(X, 3)

            beta_linear = np.polyfit(X.flatten(), target, 1)
            yhat_linear = np.polyval(beta_linear, X.flatten())
            sse_linear = np.sum((target - yhat_linear)**2)

            X_poly = np.column_stack((X.flatten(), X2.flatten(), X3.flatten()))
            beta_poly = np.linalg.lstsq(X_poly, target, rcond=None)[0]
            yhat_poly = X_poly @ beta_poly
            sse_poly = np.sum((target - yhat_poly)**2)

            if sse_poly < 0.8 * sse_linear:  # 20% improvement threshold
                return False
                
            return True
            
        except Exception:
            return True  # Default to assuming linearity if test fails
    
    @staticmethod
    def _check_homoscedasticity(feature: pd.Series, target: pd.Series) -> bool:
        """Check homoscedasticity using Spearman correlation of residuals with feature."""
        try:

            coeffs = np.polyfit(feature, target, 1)
            predicted = np.polyval(coeffs, feature)
            residuals = target - predicted

            corr, p_value = stats.spearmanr(feature, np.abs(residuals))

            return p_value > 0.05
            
        except Exception:
            return True  # Default to assuming homoscedasticity if test fails
    
    @staticmethod
    def _detect_categorical_columns(data: pd.DataFrame) -> List[str]:
        """Automatically detect categorical columns in the dataset."""
        categorical_cols = []
        
        for column in data.columns:

            if data[column].isnull().sum() / len(data) > 0.5:
                continue

            if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                categorical_cols.append(column)
            elif data[column].dtype in [np.int64, np.int32]:

                unique_vals = data[column].nunique()
                if unique_vals < min(10, len(data) * 0.05):  # 10 or 5% of rows, whichever is smaller
                    categorical_cols.append(column)
        
        return categorical_cols
    
    @staticmethod
    def _is_classification_problem(target: pd.Series) -> bool:
        """Determine if target variable represents a classification problem."""

        if target.dtype == 'object' or target.dtype.name == 'category':
            return True

        unique_count = target.nunique()

        if unique_count <= 10:
            return True

        if unique_count > min(100, len(target) * 0.1) and pd.api.types.is_numeric_dtype(target):
            return False


        value_counts = target.value_counts(normalize=True)

        if value_counts.iloc[0] > 0.3:  # Any class represents > 30% of data
            return True

        return False
    
    @staticmethod
    def _is_categorical_feature(feature: pd.Series) -> bool:
        """Determine if a feature should be treated as categorical."""
        if feature.dtype == 'object' or feature.dtype.name == 'category':
            return True

        unique_count = feature.nunique()

        if unique_count <= min(10, len(feature) * 0.05):
            return True
            
        return False 