"""Feature generation module for creating and transforming features."""

from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass
import logging
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats, signal, special

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
    performance_gain: Optional[float] = None  # Improvement in predictive performance from this feature

class FeatureGenerator:
    """Generates and transforms features based on domain knowledge."""
    
    def __init__(self, max_features_per_type: int = 3):
        """Initialize the feature generator.
        
        Args:
            max_features_per_type: Maximum number of features to generate for each transformation type
        """
        self.transformations: Dict[str, TransformationMetadata] = {}
        self._scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self._text_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.max_features_per_type = max_features_per_type
    
    def generate_features(
        self,
        data: pd.DataFrame,
        domain_insights: List[DomainInsight],
        target_column: Optional[str] = None,
        is_classification: bool = True,
    ) -> pd.DataFrame:
        """Generate new features based on domain insights.
        
        Args:
            data: Input DataFrame
            domain_insights: List of domain insights about features
            target_column: Optional name of target variable to exclude from transformations
            is_classification: Whether this is a classification task
            
        Returns:
            DataFrame with generated features
        """
        logger.info("Starting feature generation process")
        result_df = data.copy()
        
        try:
            # Create feature importance dictionary from domain insights
            feature_importance = {insight.feature_name: insight.importance 
                                 for insight in domain_insights}
            
            # Sort features by domain-based importance
            sorted_features = sorted(
                [(feat, importance) for feat, importance in feature_importance.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Prioritize the most important features first (limit to top 10)
            top_features = [f[0] for f in sorted_features[:min(10, len(sorted_features))]]
            logger.info(f"Top features by domain importance: {top_features}")
            
            # Extract transformations recommended by domain knowledge
            recommended_transforms = self._extract_recommended_transformations(domain_insights)
            logger.info(f"Recommended transformations: {recommended_transforms}")
            
            # Apply basic transformations selectively based on domain importance
            result_df = self._apply_basic_transformations(
                result_df, 
                domain_insights, 
                target_column, 
                top_features
            )
            
            # Generate interaction features based on domain knowledge
            result_df = self._generate_interaction_features(
                result_df, 
                domain_insights,
                target_column,
                is_classification
            )
            
            # Apply domain-specific transformations
            result_df = self._apply_domain_transformations(
                result_df, 
                domain_insights,
                recommended_transforms
            )
            
            # Generate text features if text columns present
            result_df = self._generate_text_features(result_df, domain_insights)
            
            # Evaluate transformation utility and keep only beneficial transformations
            if target_column:
                result_df = self._evaluate_transformations(
                    result_df,
                    data,
                    target_column,
                    is_classification
                )
            
            logger.info(f"Generated {len(result_df.columns) - len(data.columns)} new features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in feature generation: {str(e)}", exc_info=True)
            raise
    
    def _extract_recommended_transformations(
        self, 
        insights: List[DomainInsight]
    ) -> Dict[str, List[str]]:
        """Extract transformations recommended by domain knowledge."""
        recommended = {}
        for insight in insights:
            recommended[insight.feature_name] = insight.suggested_transformations
        return recommended
    
    def _apply_basic_transformations(
        self,
        data: pd.DataFrame,
        insights: List[DomainInsight],
        target_column: Optional[str],
        top_features: List[str],
    ) -> pd.DataFrame:
        """Apply basic numerical transformations selectively based on domain importance."""
        result_df = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if target_column:
            numeric_cols = [col for col in numeric_cols if col != target_column]
        
        # Create mapping of features to suggested transformations
        feature_to_transforms = {}
        for insight in insights:
            feature_to_transforms[insight.feature_name] = set(insight.suggested_transformations)
        
        for col in numeric_cols:
            # Apply transformations preferentially to top features
            is_top_feature = col in top_features
            
            # Get recommended transformations for this feature
            recommended = feature_to_transforms.get(col, set())
            
            # Log transformation for skewed features
            if (self._should_apply_log_transform(data[col]) and 
                (is_top_feature or 'log' in recommended)):
                # Ensure all values are valid for log1p (no negative values)
                valid_data = data[col].copy()
                if valid_data.min() < 0:
                    logger.warning(f"Column {col} contains negative values. Adjusting before log transformation.")
                    # Option 1: Add the absolute minimum plus a small constant to make all values positive
                    offset = abs(valid_data.min()) + 1e-6
                    valid_data = valid_data + offset
                    transform_note = f" (offset: +{offset:.6f})"
                else:
                    transform_note = ""
                
                result_df[f"log_{col}"] = np.log1p(valid_data)
                self.transformations[f"log_{col}"] = TransformationMetadata(
                    original_features=[col],
                    transformation_type="log",
                    parameters={"offset": offset if valid_data.min() < 0 else 0},
                    rationale=("High skewness detected" if 'log' not in recommended else 
                             "Log transform suggested by domain knowledge") + transform_note,
                )
            
            # Standard scaling primarily for top features or when explicitly recommended
            if is_top_feature or 'zscore' in recommended or 'scale' in recommended:
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
        target_column: Optional[str] = None,
        is_classification: bool = True,
    ) -> pd.DataFrame:
        """Generate interaction features between related variables based on domain knowledge."""
        result_df = data.copy()
        
        # Track interactions already created to avoid duplicates
        created_interactions = set()
        
        # Group insights by importance
        sorted_insights = sorted(insights, key=lambda x: x.importance, reverse=True)
        
        # Generate pairwise feature interactions based on domain relationships
        for insight in sorted_insights:
            # Only process features actually in the dataset
            if insight.feature_name not in data.columns:
                continue
                
            # Skip target column
            if target_column and insight.feature_name == target_column:
                continue
                
            # Find valid relationships (features actually in the dataset)
            valid_relationships = [
                rel for rel in insight.relationships 
                if rel in data.columns and (not target_column or rel != target_column)
            ]
            
            # Handle multiplication between related features
            if ("multiply" in insight.suggested_transformations or
                "interaction" in insight.suggested_transformations) and valid_relationships:
                
                # Limit number of interactions to prevent explosion
                for related in valid_relationships[:min(3, len(valid_relationships))]:
                    interaction_name = f"multiply_{insight.feature_name}_{related}"
                    reverse_name = f"multiply_{related}_{insight.feature_name}"
                    
                    # Check if we've already created this interaction
                    if interaction_name in created_interactions or reverse_name in created_interactions:
                        continue
                        
                    # Create the interaction feature
                    result_df[interaction_name] = data[insight.feature_name] * data[related]
                    created_interactions.add(interaction_name)
                    
                    self.transformations[interaction_name] = TransformationMetadata(
                        original_features=[insight.feature_name, related],
                        transformation_type="multiply",
                        parameters={},
                        rationale=f"Multiplication suggested by domain knowledge",
                    )
        
        # Generate polynomial interactions for top insights if polynomial transformation suggested
        for insight in sorted_insights[:5]:  # Only consider top 5 insights
            if insight.feature_name not in data.columns:
                continue
                
            if any(t.startswith('poly') for t in insight.suggested_transformations):
                # Generate polynomial features for this feature with its relations
                poly_features = [insight.feature_name] + [
                    rel for rel in insight.relationships[:2]  # Limit to top 2 relationships
                    if rel in data.columns and (not target_column or rel != target_column)
                ]
                
                if len(poly_features) > 1:  # Only if we have at least 2 features
                    # Create polynomial features
                    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                    poly_data = poly.fit_transform(data[poly_features])
                    
                    # Get feature names
                    poly_feature_names = poly.get_feature_names_out(poly_features)
                    
                    # Add polynomial features, skipping original features
                    for i, name in enumerate(poly_feature_names):
                        if ' ' in name:  # This is an interaction term
                            new_name = f"poly_{name.replace(' ', '_')}"
                            if new_name not in result_df.columns:
                                result_df[new_name] = poly_data[:, i]
                                self.transformations[new_name] = TransformationMetadata(
                                    original_features=name.split(),
                                    transformation_type="polynomial",
                                    parameters={"degree": 2},
                                    rationale="Polynomial interaction suggested by domain knowledge",
                                )
        
        return result_df
    
    def _apply_domain_transformations(
        self,
        data: pd.DataFrame,
        insights: List[DomainInsight],
        recommended_transforms: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Apply transformations suggested by domain knowledge.
        
        Args:
            data: Input DataFrame
            insights: List of domain insights about features
            recommended_transforms: Dictionary mapping features to recommended transformations
            
        Returns:
            DataFrame with domain-specific transformations applied
        """
        result_df = data.copy()
        logger.info("Starting domain transformations")
        
        # Counters to limit the number of each type of transformation
        transform_counts = {}
        
        for insight in insights:
            feature = insight.feature_name
            if feature not in data.columns:
                logger.warning(f"Feature {feature} not found in dataset")
                continue
            
            logger.info(f"Processing feature: {feature}")
            logger.info(f"Suggested transformations: {insight.suggested_transformations}")
            
            # Track features already transformed
            transformed_features = set()
            
            # Only apply transformations if they're explicitly suggested by domain knowledge
            transforms_to_apply = recommended_transforms.get(feature, [])
            
            # Apply each transformation (limiting the number of each type)
            for transform in transforms_to_apply:
                transform_type = transform.split('=')[0].strip()
                
                # Skip if we've already reached the maximum for this transform type
                if transform_counts.get(transform_type, 0) >= self.max_features_per_type:
                    continue
                
                # Update counter
                transform_counts[transform_type] = transform_counts.get(transform_type, 0) + 1
                
                # Apply the transformation
                try:
                    if transform.startswith("log") and feature not in transformed_features:
                        # Log transformation (if data is positive)
                        if data[feature].min() > 0:
                            new_feature_name = f"log_{feature}"
                            result_df[new_feature_name] = np.log(data[feature])
                            transformed_features.add(feature)
                            self.transformations[new_feature_name] = TransformationMetadata(
                                original_features=[feature],
                                transformation_type="log",
                                parameters={},
                                rationale=insight.rationale,
                            )
                            logger.info(f"Applied log transform to {feature}")
                        else:
                            # Log1p for data that includes zeros
                            new_feature_name = f"log1p_{feature}"
                            
                            # Check for negative values and adjust if needed
                            valid_data = data[feature].copy()
                            offset = 0
                            if valid_data.min() < 0:
                                logger.warning(f"Column {feature} contains negative values. Adjusting before log1p transformation.")
                                # Add the absolute minimum plus a small constant to make all values positive
                                offset = abs(valid_data.min()) + 1e-6
                                valid_data = valid_data + offset
                                transform_note = f" (offset: +{offset:.6f})"
                            else:
                                transform_note = ""
                            
                            result_df[new_feature_name] = np.log1p(valid_data)
                            transformed_features.add(feature)
                            self.transformations[new_feature_name] = TransformationMetadata(
                                original_features=[feature],
                                transformation_type="log1p",
                                parameters={"offset": offset},
                                rationale=insight.rationale + transform_note,
                            )
                            logger.info(f"Applied log1p transform to {feature}")
                    
                    elif transform.startswith("sqrt") and feature not in transformed_features:
                        # Square root transformation
                        if data[feature].min() >= 0:
                            new_feature_name = f"sqrt_{feature}"
                            result_df[new_feature_name] = np.sqrt(data[feature])
                            transformed_features.add(feature)
                            self.transformations[new_feature_name] = TransformationMetadata(
                                original_features=[feature],
                                transformation_type="sqrt",
                                parameters={},
                                rationale=insight.rationale,
                            )
                            logger.info(f"Applied sqrt transform to {feature}")
                    
                    elif transform.startswith("square") and feature not in transformed_features:
                        # Square transformation
                        new_feature_name = f"square_{feature}"
                        result_df[new_feature_name] = np.power(data[feature], 2)
                        transformed_features.add(feature)
                        self.transformations[new_feature_name] = TransformationMetadata(
                            original_features=[feature],
                            transformation_type="square",
                            parameters={},
                            rationale=insight.rationale,
                        )
                        logger.info(f"Applied square transform to {feature}")
                    
                    elif transform.startswith("cube") and feature not in transformed_features:
                        # Cube transformation
                        new_feature_name = f"cube_{feature}"
                        result_df[new_feature_name] = np.power(data[feature], 3)
                        transformed_features.add(feature)
                        self.transformations[new_feature_name] = TransformationMetadata(
                            original_features=[feature],
                            transformation_type="cube",
                            parameters={},
                            rationale=insight.rationale,
                        )
                        logger.info(f"Applied cube transform to {feature}")
                    
                    elif transform.startswith("bin") and feature not in transformed_features:
                        # Binning transformation
                        n_bins = 5  # Default number of bins
                        if "bins=" in transform:
                            try:
                                n_bins = int(transform.split("bins=")[1].split()[0])
                            except (IndexError, ValueError):
                                pass
                        
                        new_feature_name = f"binned_{feature}"
                        
                        try:
                            # Use the more robust binning method
                            result_df[new_feature_name] = self._apply_binning(data[feature], n_bins=n_bins)
                            transformed_features.add(feature)
                            
                            self.transformations[new_feature_name] = TransformationMetadata(
                                original_features=[feature],
                                transformation_type="binning",
                                parameters={"n_bins": n_bins},
                                rationale=insight.rationale,
                            )
                            logger.info(f"Applied binning transform to {feature}")
                        except Exception as e:
                            logger.warning(f"Error applying binning to {feature}: {str(e)}")
                    
                    elif transform.startswith("zscore") and feature not in transformed_features:
                        # Z-score normalization
                        new_feature_name = f"zscore_{feature}"
                        result_df[new_feature_name] = (data[feature] - data[feature].mean()) / data[feature].std()
                        transformed_features.add(feature)
                        self.transformations[new_feature_name] = TransformationMetadata(
                            original_features=[feature],
                            transformation_type="zscore",
                            parameters={},
                            rationale=insight.rationale,
                        )
                        logger.info(f"Applied z-score transform to {feature}")
                    
                    elif transform.startswith("minmax") and feature not in transformed_features:
                        # Min-max scaling
                        new_feature_name = f"minmax_{feature}"
                        result_df[new_feature_name] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
                        transformed_features.add(feature)
                        self.transformations[new_feature_name] = TransformationMetadata(
                            original_features=[feature],
                            transformation_type="minmax",
                            parameters={},
                            rationale=insight.rationale,
                        )
                        logger.info(f"Applied min-max transform to {feature}")
                    
                    elif transform.startswith("one_hot") and feature not in transformed_features:
                        # One-hot encoding for categorical features
                        try:
                            # Use the more robust one-hot encoding method
                            encoded_df = self._apply_one_hot_encoding(data[feature])
                            
                            # Only proceed if we got some encoded columns
                            if not encoded_df.empty:
                                # Add each encoded column to the result
                                for col in encoded_df.columns:
                                    result_df[col] = encoded_df[col]
                                    self.transformations[col] = TransformationMetadata(
                                        original_features=[feature],
                                        transformation_type="one_hot",
                                        parameters={},
                                        rationale=insight.rationale,
                                    )
                                transformed_features.add(feature)
                                logger.info(f"Applied one-hot encoding to {feature}")
                            else:
                                logger.warning(f"One-hot encoding produced no columns for {feature}")
                        except Exception as e:
                            logger.warning(f"Error applying one-hot encoding to {feature}: {str(e)}")
                    
                    # scipy.stats transformations
                    elif transform.startswith("boxcox") and feature not in transformed_features:
                        # Box-Cox transformation for positive data
                        if data[feature].min() > 0:
                            new_feature_name = f"boxcox_{feature}"
                            result_df[new_feature_name], _ = stats.boxcox(data[feature])
                            transformed_features.add(feature)
                            self.transformations[new_feature_name] = TransformationMetadata(
                                original_features=[feature],
                                transformation_type="boxcox",
                                parameters={},
                                rationale=insight.rationale,
                            )
                            logger.info(f"Applied Box-Cox transform to {feature}")
                    
                    elif transform.startswith("yeojohnson") and feature not in transformed_features:
                        # Yeo-Johnson transformation (works with negative values)
                        new_feature_name = f"yeojohnson_{feature}"
                        result_df[new_feature_name], _ = stats.yeojohnson(data[feature])
                        transformed_features.add(feature)
                        self.transformations[new_feature_name] = TransformationMetadata(
                            original_features=[feature],
                            transformation_type="yeojohnson",
                            parameters={},
                            rationale=insight.rationale,
                        )
                        logger.info(f"Applied Yeo-Johnson transform to {feature}")
                    
                    elif transform.startswith("quantile") and feature not in transformed_features:
                        # Quantile transformation to normal distribution
                        new_feature_name = f"quantile_{feature}"
                        result_df[new_feature_name] = stats.norm.ppf(
                            stats.rankdata(data[feature]) / (len(data[feature]) + 1)
                        )
                        transformed_features.add(feature)
                        self.transformations[new_feature_name] = TransformationMetadata(
                            original_features=[feature],
                            transformation_type="quantile",
                            parameters={},
                            rationale=insight.rationale,
                        )
                        logger.info(f"Applied quantile transform to {feature}")
                
                except Exception as e:
                    logger.warning(f"Failed to apply {transform} to {feature}: {str(e)}")
        
        logger.info(f"Final columns after transformations: {result_df.columns.tolist()}")
        logger.info(f"Recorded transformations: {list(self.transformations.keys())}")
        return result_df
    
    def _generate_text_features(
        self,
        data: pd.DataFrame,
        insights: List[DomainInsight],
    ) -> pd.DataFrame:
        """Generate features from text columns."""
        result_df = data.copy()
        text_cols = data.select_dtypes(include=['object']).columns
        
        # Get text columns specifically mentioned in domain insights
        text_insights = [insight for insight in insights 
                         if insight.feature_name in text_cols and 
                         any(t.startswith("text") or t.startswith("tfidf") or t.startswith("nlp") 
                             for t in insight.suggested_transformations)]
        
        # If no specific text columns are highlighted, check all potential text columns
        if not text_insights:
            for col in text_cols:
                if self._is_text_column(data[col]):
                    vectorizer = TfidfVectorizer(max_features=5)  # Reduced from 10 to 5 features
                    try:
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
                    except Exception as e:
                        logger.warning(f"Failed to vectorize text column {col}: {str(e)}")
        else:
            # Process only text columns mentioned in domain insights
            for insight in text_insights:
                col = insight.feature_name
                vectorizer = TfidfVectorizer(max_features=7)
                try:
                    text_features = vectorizer.fit_transform(data[col].fillna(''))
                    feature_names = [f"tfidf_{col}_{i}" for i in range(text_features.shape[1])]
                    
                    for i, name in enumerate(feature_names):
                        result_df[name] = text_features[:, i].toarray()
                        self.transformations[name] = TransformationMetadata(
                            original_features=[col],
                            transformation_type="tfidf",
                            parameters={"index": i},
                            rationale=insight.rationale,
                        )
                    
                    self._text_vectorizers[col] = vectorizer
                except Exception as e:
                    logger.warning(f"Failed to vectorize text column {col}: {str(e)}")
        
        return result_df
    
    def _evaluate_transformations(
        self, 
        transformed_data: pd.DataFrame,
        original_data: pd.DataFrame,
        target_column: str,
        is_classification: bool = True,
    ) -> pd.DataFrame:
        """Evaluate the utility of transformations using XGBoost with anti-overfitting techniques."""
        try:
            # Group transformations by type for evaluation
            transform_groups = {}
            for col in transformed_data.columns:
                if col in self.transformations:
                    transform_type = self.transformations[col].transformation_type
                    if transform_type not in transform_groups:
                        transform_groups[transform_type] = []
                    transform_groups[transform_type].append(col)
            
            if not transform_groups:
                logger.info("No transformations to evaluate")
                return transformed_data
            
            # Import XGBoost
            import xgboost as xgb
            
            # Prepare data for evaluation
            X = original_data.drop(columns=[target_column])
            y = original_data[target_column]
            
            # Use stratified sampling to preserve class distribution
            from sklearn.model_selection import train_test_split
            if is_classification:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
            
            # Select evaluation model and metrics based on problem type
            if is_classification:
                if len(np.unique(y)) == 2:
                    scoring_func = roc_auc_score
                    model_class = xgb.XGBClassifier
                    model_params = {
                        'n_estimators': 50,
                        'learning_rate': 0.05,
                        'max_depth': 3,
                        'min_child_weight': 2,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'gamma': 1,
                        'reg_alpha': 0.1,
                        'reg_lambda': 1,
                        'random_state': 42
                    }
                else:
                    scoring_func = accuracy_score
                    model_class = xgb.XGBClassifier
                    model_params = {
                        'n_estimators': 50,
                        'learning_rate': 0.05,
                        'max_depth': 3,
                        'min_child_weight': 2,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'gamma': 1,
                        'reg_alpha': 0.1,
                        'reg_lambda': 1,
                        'random_state': 42
                    }
            else:
                scoring_func = r2_score
                model_class = xgb.XGBRegressor
                model_params = {
                    'n_estimators': 50,
                    'learning_rate': 0.05,
                    'max_depth': 3,
                    'min_child_weight': 2,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1,
                    'random_state': 42
                }
            
            # Train baseline model on original features with early stopping
            baseline_model = model_class(**model_params)
            
            # Handle potential issues in validation data
            X_val_orig = X.iloc[X_val.index].copy()
            eval_metric = 'auc' if is_classification and len(np.unique(y)) == 2 else ('error' if is_classification else 'rmse')
            
            # Fit with early stopping to prevent overfitting
            try:
                # First attempt using newer XGBoost API
                baseline_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val_orig, y_val)],
                    early_stopping_rounds=5,
                    eval_metric=eval_metric,
                    verbose=False
                )
            except TypeError as e:
                if "early_stopping_rounds" in str(e):
                    logger.warning("XGBoost API doesn't support early_stopping_rounds parameter in fit(), using alternative approach")
                    # Fallback to older XGBoost API or modified approach
                    baseline_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val_orig, y_val)],
                        verbose=False
                    )
                else:
                    # Some other TypeError
                    logger.error(f"XGBoost fit error: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Error in baseline model fitting: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Fallback to simpler model without validation
                baseline_model.fit(X_train, y_train)
            
            # Get baseline performance
            if is_classification and len(np.unique(y)) == 2:
                y_prob = baseline_model.predict_proba(X_val_orig)[:, 1]
                baseline_score = scoring_func(y_val, y_prob)
            else:
                y_pred = baseline_model.predict(X_val_orig)
                baseline_score = scoring_func(y_val, y_pred)
            
            logger.info(f"Baseline model performance: {baseline_score:.4f}")
            
            # Keep track of features from original data and beneficial transformations
            beneficial_features = list(original_data.columns)
            feature_gains = {}
            
            # Store performance improvement by transformation type
            transform_performance = {}
            
            # Evaluate each transformation type
            for transform_type, cols in transform_groups.items():
                logger.info(f"Evaluating {transform_type} transformations ({len(cols)} features)")
                
                # Create model for this transformation type
                eval_model = model_class(**model_params)
                
                # Get transformed data with these specific transformed features + original features
                X_trans = pd.concat([
                    X,  # original features
                    transformed_data[cols]  # only the current transformation group
                ], axis=1)
                
                X_train_trans = X_trans.iloc[X_train.index]
                X_val_trans = X_trans.iloc[X_val.index]
                
                # Fit model with early stopping
                try:
                    # First attempt using newer XGBoost API
                    eval_model.fit(
                        X_train_trans, y_train,
                        eval_set=[(X_val_trans, y_val)],
                        early_stopping_rounds=5,
                        eval_metric=eval_metric,
                        verbose=False
                    )
                except TypeError as e:
                    if "early_stopping_rounds" in str(e):
                        logger.warning("XGBoost API doesn't support early_stopping_rounds parameter in fit(), using alternative approach")
                        # Fallback to older XGBoost API or modified approach
                        eval_model.fit(
                            X_train_trans, y_train,
                            eval_set=[(X_val_trans, y_val)],
                            verbose=False
                        )
                    else:
                        # Some other TypeError
                        logger.error(f"XGBoost fit error: {str(e)}")
                        raise
                except Exception as e:
                    logger.error(f"Error in transformation model fitting: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Fallback to simpler model without validation
                    eval_model.fit(X_train_trans, y_train)
                
                # Evaluate performance
                try:
                    if is_classification and len(np.unique(y)) == 2:
                        y_prob = eval_model.predict_proba(X_val_trans)[:, 1]
                        score = scoring_func(y_val, y_prob)
                    else:
                        y_pred = eval_model.predict(X_val_trans)
                        score = scoring_func(y_val, y_pred)
                    
                    # Calculate improvement
                    improvement = score - baseline_score
                    transform_performance[transform_type] = improvement
                    logger.info(f"{transform_type} transformation performance: {score:.4f} (improvement: {improvement:.4f})")
                except Exception as e:
                    logger.error(f"Error evaluating transformation performance: {str(e)}")
                    # Use a conservative approach - assume no improvement
                    improvement = -0.01
                    transform_performance[transform_type] = improvement
                    logger.warning(f"Using fallback score for {transform_type} transformations")
                
                # Keep transformations that don't degrade performance
                # Use a small negative threshold to accommodate for randomness
                if improvement >= -0.01:  # Allow slight performance degradation due to randomness
                    beneficial_features.extend(cols)
                    
                    # Get feature importance for these transformations
                    try:
                        importances = eval_model.feature_importances_
                        feature_names = list(X_train_trans.columns)
                        
                        # Record importance for the transformed features
                        for col in cols:
                            if col in feature_names:
                                col_idx = feature_names.index(col)
                                imp_value = importances[col_idx]
                                feature_gains[col] = imp_value
                    except Exception as e:
                        logger.error(f"Error getting feature importances: {str(e)}")
                        # Set default importance values
                        for col in cols:
                            feature_gains[col] = 0.01  # Small default value
            
            # Output performance by transformation type
            logger.info("Performance improvement by transformation type:")
            for t_type, improvement in sorted(transform_performance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {t_type}: {improvement:.4f}")
            
            # Keep only beneficial features from the transformed data
            beneficial_features = list(set(beneficial_features))  # Remove duplicates
            result_df = transformed_data[beneficial_features].copy()
            
            # Log removed features
            removed_features = set(transformed_data.columns) - set(beneficial_features)
            if removed_features:
                logger.info(f"Removed {len(removed_features)} non-beneficial transformed features")
                logger.debug(f"Removed features: {removed_features}")
                
            return result_df
        
        except Exception as e:
            logger.error(f"Error in transformation evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            # If evaluation fails, return the original transformed data
            return transformed_data
    
    @staticmethod
    def _should_apply_log_transform(series: pd.Series) -> bool:
        """Check if log transform should be applied based on skewness."""
        return abs(series.skew()) > 1.0 and series.min() >= 0
    
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
        if len(sample) == 0:
            return False
        # Consider it text if at least 20% of values have more than 3 words
        text_values = [x for x in sample if isinstance(x, str) and len(x.split()) > 3]
        return len(text_values) >= max(1, 0.2 * len(sample))
    
    def _apply_binning(self, series: pd.Series, n_bins: int = 5) -> pd.Series:
        """Apply equal-width binning to a numerical feature."""
        try:
            # Handle non-numeric data
            if not pd.api.types.is_numeric_dtype(series):
                logger.warning(f"Cannot apply binning to non-numeric column: {series.name}")
                return series
                
            # Create bins using pandas cut
            binned = pd.cut(
                series,
                bins=n_bins,
                labels=False,
                include_lowest=True,
                duplicates='drop'
            )
            
            # Handle potential NaN values
            if binned.isna().any():
                logger.warning(f"Binning produced NaN values for column {series.name}")
                # Replace NaNs with most frequent bin
                mode_bin = binned.mode().iloc[0] if not binned.dropna().empty else 0
                binned = binned.fillna(mode_bin)
                
            return binned
            
        except Exception as e:
            logger.warning(f"Error applying binning to {series.name}: {str(e)}")
            return series
            
    def _apply_one_hot_encoding(self, series: pd.Series) -> pd.DataFrame:
        """Apply one-hot encoding to a categorical feature."""
        try:
            # For numeric columns, convert to string first to treat as categorical
            if pd.api.types.is_numeric_dtype(series):
                series = series.astype(str)
                
            # Use pandas get_dummies
            encoded = pd.get_dummies(series, prefix=f"onehot_{series.name}")
            
            return encoded
            
        except Exception as e:
            logger.warning(f"Error applying one-hot encoding to {series.name}: {str(e)}")
            # Return empty DataFrame on error
            return pd.DataFrame(index=series.index) 