"""Pipeline module for orchestrating the feature engineering process."""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import time
import uuid
import datetime
import traceback

import pandas as pd
from pydantic import BaseModel
import numpy as np

from faster.domain_knowledge import DomainKnowledgeExtractor
from faster.feature_generation import FeatureGenerator, TransformationMetadata
from faster.statistical_evaluation import StatisticalEvaluator, FeatureStatistics
from faster.feature_selection import FeatureSelector, SelectionCriteria, SelectionResult
from faster.utils.logging import get_logger, setup_logging
from faster.utils.validation import validate_dataframe

logger = get_logger(__name__)

class PipelineConfig(BaseModel):
    """Configuration for the FASTER pipeline."""
    
    # LLM settings
    model_name: str = "deepseek/deepseek-chat:free"
    temperature: float = 0.0
    
    # Feature generation settings
    max_interaction_degree: int = 2
    text_feature_method: str = "tfidf"
    
    # Statistical evaluation settings
    alpha: float = 0.05
    correction_method: str = "fdr_bh"
    
    # Feature selection settings
    selection_criteria: SelectionCriteria = SelectionCriteria()
    keep_all_features: bool = False
    
    # Output settings
    output_dir: Optional[str] = None
    save_intermediate: bool = False

@dataclass
class PipelineResult:
    """Results from the FASTER pipeline."""
    
    transformed_data: pd.DataFrame
    selected_features: List[str]
    feature_metadata: Dict[str, Dict]
    performance_metrics: Dict[str, float]
    execution_log: Dict[str, Dict]

class Pipeline:
    """Main pipeline for automated feature engineering."""
    
    def __init__(
        self,
        config: Optional[Union[PipelineConfig, Dict]] = None,
    ):
        """Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = (
            config if isinstance(config, PipelineConfig)
            else PipelineConfig(**(config or {}))
        )
        
        # Initialize components
        self.domain_extractor = DomainKnowledgeExtractor(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
        )
        
        self.feature_generator = FeatureGenerator()
        
        self.statistical_evaluator = StatisticalEvaluator(
            alpha=self.config.alpha,
            correction_method=self.config.correction_method,
        )
        
        # Create selection criteria with keep_all_features value from config
        selection_criteria = self.config.selection_criteria
        selection_criteria.keep_all_features = self.config.keep_all_features
        
        self.feature_selector = FeatureSelector(
            criteria=selection_criteria,
        )
        
        # Setup logging
        if self.config.output_dir:
            setup_logging(Path(self.config.output_dir) / "faster.log")
    
    def run(
        self,
        data: pd.DataFrame,
        target_column: str,
        problem_description: str,
        categorical_columns: Optional[List[str]] = None,
        domain_context: Optional[str] = None,
        is_classification: bool = True,
        keep_all_features: bool = False,
    ) -> PipelineResult:
        """Run the full FASTER pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Column name of the target variable
            problem_description: Description of the problem
            categorical_columns: Columns to treat as categorical
            domain_context: Additional domain context
            is_classification: Whether this is a classification task
            keep_all_features: Whether to keep all features without applying selection filters
            
        Returns:
            PipelineResult with transformed data and metadata
        """
        # Validate input data
        validate_dataframe(data, target_column)
        
        # Store original state to allow fallback
        original_data = data.copy()
        
        # Setup logging and execution tracking
        run_id = str(uuid.uuid4())
        execution_log = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "input_shape": data.shape,
            "target_column": target_column,
            "is_classification": is_classification,
            "steps": {},
        }
        
        try:
            # 1. Extract domain knowledge
            step_start = time.time()
            
            # Temporarily store the current keep_all_features value
            original_keep_all_features = self.feature_selector.criteria.keep_all_features
            
            # Update the keep_all_features value based on the parameter
            self.feature_selector.criteria.keep_all_features = keep_all_features
            
            try:
                domain_insights = self.domain_extractor.extract_domain_knowledge(
                    problem_description=problem_description,
                    data_sample=data.head(5),
                    column_names=list(data.columns),
                    categorical_columns=categorical_columns,
                    domain_context=domain_context,
                )
                
                execution_log["steps"]["domain_knowledge"] = {
                    "duration": time.time() - step_start,
                    "insights_extracted": len(domain_insights),
                }
                
            except Exception as e:
                logger.error(f"Error in domain knowledge extraction: {str(e)}")
                logger.warning("Proceeding without domain knowledge")
                domain_insights = []
                
                execution_log["steps"]["domain_knowledge"] = {
                    "duration": time.time() - step_start,
                    "status": "error",
                    "error": str(e),
                }
                
            # 2. Generate features
            step_start = time.time()
            try:
                transformed_data = self.feature_generator.generate_features(
                    data=data,
                    target_column=target_column,
                    categorical_columns=categorical_columns,
                    domain_insights=domain_insights,
                    max_interaction_degree=self.config.max_interaction_degree,
                    text_feature_method=self.config.text_feature_method,
                )
                
                execution_log["steps"]["feature_generation"] = {
                    "duration": time.time() - step_start,
                    "features_generated": transformed_data.shape[1] - data.shape[1],
                }
                
            except Exception as e:
                logger.error(f"Error in feature generation: {str(e)}")
                logger.warning("Using original features as fallback")
                transformed_data = data.copy()
                
                execution_log["steps"]["feature_generation"] = {
                    "duration": time.time() - step_start,
                    "status": "error",
                    "error": str(e),
                }
            
            # 3. Evaluate feature statistics
            step_start = time.time()
            try:
                feature_stats = self.statistical_evaluator.evaluate_features(
                    data=transformed_data,
                    target_column=target_column,
                    categorical_columns=categorical_columns,
                )
                
                execution_log["steps"]["statistical_evaluation"] = {
                    "duration": time.time() - step_start,
                    "features_evaluated": len(feature_stats),
                }
                
            except Exception as e:
                logger.error(f"Error in statistical evaluation: {str(e)}")
                logger.warning("Proceeding with limited statistical information")
                feature_stats = []
                
                execution_log["steps"]["statistical_evaluation"] = {
                    "duration": time.time() - step_start,
                    "status": "error",
                    "error": str(e),
                }
            
            # 4. Select features
            step_start = time.time()
            
            try:
                selection_result = self.feature_selector.select_features(
                    data=transformed_data,
                    target_column=target_column,
                    feature_stats=feature_stats,
                    is_classification=is_classification,
                )
                
                # Compile selected features into final dataset
                selected_columns = selection_result.selected_features + [target_column]
                final_data = transformed_data[selected_columns].copy()
                
                execution_log["steps"]["feature_selection"] = {
                    "duration": time.time() - step_start,
                    "selected_features": len(selection_result.selected_features),
                    "removed_features": len(transformed_data.columns) - len(selected_columns),
                    "keep_all_features": keep_all_features
                }
                
                # Gather feature metadata
                feature_metadata = self._compile_feature_metadata(
                    selection_result=selection_result,
                    transformations=self.feature_generator.transformations,
                )
                
            except Exception as e:
                logger.error(f"Error in feature selection: {str(e)}")
                logger.warning("Using all generated features as fallback")
                # Fallback: use all features
                all_features = [col for col in transformed_data.columns if col != target_column]
                selection_result = SelectionResult(
                    selected_features=all_features,
                    selection_scores={f: 1.0 for f in all_features},
                    removed_features={},
                    statistics={},
                )
                selected_columns = all_features + [target_column]
                final_data = transformed_data[selected_columns].copy()
                feature_metadata = {f: {} for f in all_features}
                
                execution_log["steps"]["feature_selection"] = {
                    "duration": time.time() - step_start,
                    "selected_features": len(all_features),
                    "removed_features": 0,
                    "status": "fallback",
                    "keep_all_features": keep_all_features
                }
            
            # 5. Calculate performance metrics
            step_start = time.time()
            
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import make_scorer, r2_score, accuracy_score, f1_score, roc_auc_score
                
                if is_classification:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    scorer = make_scorer(accuracy_score)
                    multi_class = len(np.unique(data[target_column])) > 2
                    
                    if multi_class:
                        auc_scorer = make_scorer(f1_score, average='weighted')
                    else:
                        auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    scorer = make_scorer(r2_score)
                    auc_scorer = None
                
                # Calculate performance for transformed features
                X = final_data.drop(columns=[target_column])
                y = final_data[target_column]
                
                cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
                performance_metrics = {
                    "mean_score": float(np.mean(cv_scores)),
                    "std_score": float(np.std(cv_scores)),
                }
                
                if auc_scorer and is_classification and not multi_class:
                    auc_scores = cross_val_score(model, X, y, cv=5, scoring=auc_scorer)
                    performance_metrics["mean_auc"] = float(np.mean(auc_scores))
                
                # Calculate baseline performance with original features
                use_transformed_features = True
                
                # Only compare if we have actually transformed features
                if transformed_data.shape[1] > data.shape[1]:
                    try:
                        # Get baseline features (no target)
                        X_baseline = original_data.drop(columns=[target_column])
                        y_baseline = original_data[target_column]
                        
                        baseline_scores = cross_val_score(model, X_baseline, y_baseline, cv=5, scoring=scorer)
                        baseline_metrics = {
                            "mean_score": float(np.mean(baseline_scores)),
                            "std_score": float(np.std(baseline_scores)),
                        }
                        
                        if auc_scorer and is_classification and not multi_class:
                            baseline_auc = cross_val_score(model, X_baseline, y_baseline, cv=5, scoring=auc_scorer)
                            baseline_metrics["mean_auc"] = float(np.mean(baseline_auc))
                        
                        # Compare performance
                        performance_diff = performance_metrics["mean_score"] - baseline_metrics["mean_score"]
                        use_transformed_features = performance_diff >= -0.02  # Allow slight degradation
                        
                        execution_log["steps"]["performance_evaluation"] = {
                            "duration": time.time() - step_start,
                            "transformed_score": performance_metrics["mean_score"],
                            "baseline_score": baseline_metrics["mean_score"],
                            "performance_diff": performance_diff,
                            "use_transformed": use_transformed_features,
                        }
                        
                    except Exception as e:
                        logger.error(f"Error in baseline comparison: {str(e)}")
                        use_transformed_features = True  # Default to using transformed features
                
                # If baseline is better, fallback to it
                if not use_transformed_features:
                    logger.warning("Baseline features outperform transformed features. Using baseline.")
                    # Use original data and recalculate selection
                    try:
                        baseline_stats = self.statistical_evaluator.evaluate_features(
                            data=original_data,
                            target_column=target_column,
                            categorical_columns=categorical_columns,
                        )
                        
                        baseline_selection = self.feature_selector.select_features(
                            data=original_data,
                            target_column=target_column,
                            feature_stats=baseline_stats,
                            is_classification=is_classification,
                        )
                        
                        # Use baseline selection
                        selected_columns = baseline_selection.selected_features + [target_column]
                        final_data = original_data[selected_columns].copy()
                        
                        # Update metadata
                        feature_metadata = {
                            f: {"source": "original", "transformation": None}
                            for f in baseline_selection.selected_features
                        }
                        
                        # Use baseline metrics
                        performance_metrics = baseline_metrics
                        
                        # Update execution log
                        execution_log["steps"]["feature_selection"]["status"] = "reverted_to_baseline"
                        execution_log["steps"]["feature_selection"]["selected_features"] = len(baseline_selection.selected_features)
                    except Exception as e:
                        logger.error(f"Error in baseline selection: {str(e)}")
                        # Continue with transformed features
                
            except Exception as e:
                logger.error(f"Error in performance evaluation: {str(e)}")
                performance_metrics = {"error": str(e)}
                
                execution_log["steps"]["performance_evaluation"] = {
                    "duration": time.time() - step_start,
                    "status": "error",
                    "error": str(e),
                }
            
            # Save intermediate results if requested
            if self.config.save_intermediate and self.config.output_dir:
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # Save execution log
                with open(output_dir / f"execution_log_{run_id}.json", "w") as f:
                    json.dump(execution_log, f, indent=2, default=str)
                
                # Save final dataset
                final_data.to_csv(output_dir / f"final_data_{run_id}.csv", index=False)
            
            # Restore the original keep_all_features value
            self.feature_selector.criteria.keep_all_features = original_keep_all_features
            
            return PipelineResult(
                transformed_data=final_data,
                selected_features=selection_result.selected_features,
                feature_metadata=feature_metadata,
                performance_metrics=performance_metrics,
                execution_log=execution_log,
            )
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            traceback.print_exc()
            
            # Create minimal result with error information
            error_log = {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            
            # Combine with existing execution log
            execution_log["error"] = error_log
            
            # Try to restore the original keep_all_features value in case of exception
            try:
                self.feature_selector.criteria.keep_all_features = original_keep_all_features
            except:
                pass
            
            return PipelineResult(
                transformed_data=data,
                selected_features=[],
                feature_metadata={},
                performance_metrics={"error": str(e)},
                execution_log=execution_log,
            )

    def _compare_metrics(self, transformed_metrics: Dict[str, float], baseline_metrics: Dict[str, float], 
                         is_classification: bool) -> bool:
        """Compare baseline and transformed metrics to determine if features improved model performance."""
        if is_classification:
            primary_metric = "test_f1"
        else:
            primary_metric = "test_r2"
        
        # Check if the primary metric is in both dictionaries
        if primary_metric not in transformed_metrics or primary_metric not in baseline_metrics:
            logger.warning(f"Primary metric {primary_metric} not found in metrics. Using first available metric.")
            # Get the first test metric available in both
            for metric in transformed_metrics:
                if metric in baseline_metrics and metric.startswith("test_"):
                    primary_metric = metric
                    break
            else:
                logger.error("No common test metrics found between baseline and transformed.")
                return False
        
        # Compare the metrics
        return transformed_metrics[primary_metric] > baseline_metrics[primary_metric]

    def _create_dataset_summary(self, data: pd.DataFrame, target_column: str, 
                               categorical_columns: Optional[List[str]] = None) -> Dict[str, any]:
        """Create a summary of the dataset for the domain knowledge extractor.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            categorical_columns: List of categorical columns
            
        Returns:
            Dictionary with dataset summary
        """
        # Identify numeric and categorical columns
        if categorical_columns is None:
            categorical_columns = []
            for col in data.columns:
                if col == target_column:
                    continue
                if pd.api.types.is_numeric_dtype(data[col]) and len(data[col].unique()) > 10:
                    pass  # Numeric column
                else:
                    categorical_columns.append(col)
        
        # Create feature descriptions
        feature_descriptions = {}
        for col in data.columns:
            if col == target_column:
                continue
                
            # Get basic stats
            if col in categorical_columns:
                value_counts = data[col].value_counts().head(5).to_dict()
                feature_descriptions[col] = {
                    "type": "categorical",
                    "unique_values": len(data[col].unique()),
                    "top_values": value_counts,
                    "missing": int(data[col].isnull().sum())
                }
            else:
                feature_descriptions[col] = {
                    "type": "numeric",
                    "min": float(data[col].min()) if not pd.isna(data[col].min()) else None,
                    "max": float(data[col].max()) if not pd.isna(data[col].max()) else None,
                    "mean": float(data[col].mean()) if not pd.isna(data[col].mean()) else None,
                    "std": float(data[col].std()) if not pd.isna(data[col].std()) else None,
                    "missing": int(data[col].isnull().sum())
                }
        
        # Get target info
        target_info = {}
        if target_column in data.columns:
            if target_column in categorical_columns:
                value_counts = data[target_column].value_counts().to_dict()
                target_info = {
                    "type": "categorical",
                    "unique_values": len(data[target_column].unique()),
                    "distribution": value_counts,
                    "missing": int(data[target_column].isnull().sum())
                }
            else:
                target_info = {
                    "type": "numeric",
                    "min": float(data[target_column].min()) if not pd.isna(data[target_column].min()) else None,
                    "max": float(data[target_column].max()) if not pd.isna(data[target_column].max()) else None,
                    "mean": float(data[target_column].mean()) if not pd.isna(data[target_column].mean()) else None,
                    "std": float(data[target_column].std()) if not pd.isna(data[target_column].std()) else None,
                    "missing": int(data[target_column].isnull().sum())
                }
        
        # Create summary
        summary = {
            "n_samples": len(data),
            "n_features": len(data.columns) - 1,
            "categorical_columns": categorical_columns,
            "target_column": target_column,
            "feature_descriptions": feature_descriptions,
            "target_info": target_info
        }
        
        return summary

    def _create_fallback_feature_stats(self, data: pd.DataFrame, target_column: str) -> List[FeatureStatistics]:
        """Create fallback feature statistics for all columns in the dataset."""
        features = [col for col in data.columns if col != target_column]
        stats = []
        
        for feature in features:
            # Create a minimal FeatureStatistics object
            stat = FeatureStatistics(
                feature_name=feature,
                correlation=None,
                p_value=0.5,  # Neutral p-value
                effect_size=0.0,
                mutual_information=0.0,
                test_method="none",
                assumptions_met={},
                warnings=["Fallback statistics due to evaluation error"],
                predictive_power=0.0
            )
            stats.append(stat)
        
        return stats
    
    def _save_intermediate(self, filename: str, data: Union[pd.DataFrame, dict]) -> None:
        """Save intermediate results."""
        if not self.config.output_dir:
            return
            
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_dir / filename, index=False)
        else:
            with open(output_dir / filename, "w") as f:
                json.dump(data, f, indent=2)
    
    def _compile_feature_metadata(
        self,
        selection_result: "SelectionResult",
        transformations: Dict[str, "TransformationMetadata"],
    ) -> Dict[str, Dict]:
        """Compile metadata for selected features."""
        metadata = {}
        
        for feature in selection_result.selected_features:
            transform_info = transformations.get(feature)
            
            # Get base feature name for statistics (original feature before transformation)
            base_feature = feature
            if transform_info:
                base_feature = transform_info.original_features[0]  # Use first original feature
            
            metadata[feature] = {
                "importance_score": selection_result.selection_scores.get(feature, 0.0),
                "statistics": selection_result.statistics.get(base_feature, {}),
                "transformation": transform_info.__dict__ if transform_info else {},
            }
        
        return metadata
    
    def _calculate_performance_metrics(
        self,
        data: pd.DataFrame,
        target_column: str,
        is_classification: bool,
    ) -> Dict[str, float]:
        """Calculate performance metrics for the selected feature set using cross-validation with XGBoost."""
        try:
            if len(data.columns) <= 1:  # Only has the target column
                logger.warning("No features available for performance evaluation")
                return {
                    "n_features": 0,
                    "memory_usage": data.memory_usage().sum() / 1024**2,  # MB
                }
            
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Basic data validation
            if X.shape[0] < 10:  # Too few samples for meaningful CV
                logger.warning("Too few samples for cross-validation")
                return {
                    "n_features": len(X.columns),
                    "memory_usage": data.memory_usage().sum() / 1024**2,  # MB
                }
            
            # Handle categorical features
            categorical_columns = []
            for col in X.columns:
                if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
                    categorical_columns.append(col)
            
            # One-hot encode categorical features
            if categorical_columns:
                X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            
            # Check for infinite or NaN values
            if X.isna().any().any() or np.isinf(X.values).any():
                logger.warning("Data contains NaN or infinite values; filling with appropriate values")
                X = X.replace([np.inf, -np.inf], np.nan)
                for col in X.columns:
                    if X[col].isna().any():
                        if pd.api.types.is_numeric_dtype(X[col]):
                            X[col] = X[col].fillna(X[col].median())
                        else:
                            X[col] = X[col].fillna(X[col].mode()[0])
            
            # Import required modules
            from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
            import xgboost as xgb
            
            # Configure XGBoost with anti-overfitting parameters
            if is_classification:
                # For small datasets, keep models simple with regularization
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,  # Shallow trees to prevent overfitting
                    min_child_weight=2,  # Higher values prevent overfitting
                    subsample=0.8,  # Use 80% of data for trees
                    colsample_bytree=0.8,  # Use 80% of features for each tree
                    gamma=1,  # Minimum loss reduction for split
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1,  # L2 regularization
                    random_state=42,
                    enable_categorical=True,
                    use_label_encoder=False
                )
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                if len(np.unique(y)) == 2:  # Binary classification
                    scoring = ['accuracy', 'roc_auc', 'f1']
                else:  # Multi-class
                    scoring = ['accuracy', 'f1_weighted']
            else:  # Regression
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=1,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=42,
                    enable_categorical=True
                )
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
            
            # Perform cross-validation
            try:
                # For checking if data has infinity or NaN values
                if np.isinf(X.values).any() or np.isnan(X.values).any():
                    logger.warning("Data contains infinite or NaN values. Cleaning before CV.")
                    # Replace inf values with NaN
                    X = X.replace([np.inf, -np.inf], np.nan)
                    # Fill NaN values with median for numeric columns
                    for col in X.columns:
                        if pd.api.types.is_numeric_dtype(X[col]) and X[col].isna().any():
                            X[col] = X[col].fillna(X[col].median())
                
                cv_results = cross_validate(
                    model, X, y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=True
                )
                
                # Calculate overfitting ratio
                overfitting_metrics = {}
                metrics_mapping = {
                    'r2': ('test_r2', 'train_r2'),
                    'neg_mean_absolute_error': ('test_neg_mean_absolute_error', 'train_neg_mean_absolute_error'),
                    'neg_root_mean_squared_error': ('test_neg_root_mean_squared_error', 'train_neg_root_mean_squared_error'),
                    'accuracy': ('test_accuracy', 'train_accuracy'),
                    'roc_auc': ('test_roc_auc', 'train_roc_auc'),
                    'f1': ('test_f1', 'train_f1'),
                    'f1_weighted': ('test_f1_weighted', 'train_f1_weighted')
                }
                
                for metric, (test_key, train_key) in metrics_mapping.items():
                    if test_key in cv_results and train_key in cv_results:
                        test_mean = np.mean(cv_results[test_key])
                        train_mean = np.mean(cv_results[train_key])
                        
                        # Avoid division by zero
                        if train_mean != 0 and not np.isnan(train_mean) and not np.isnan(test_mean):
                            if metric.startswith('neg_'):  # For metrics where lower is better
                                # Use absolute values for negative metrics
                                overfitting_metrics[f"{metric}_overfit_ratio"] = abs(test_mean) / abs(train_mean)
                            else:  # For metrics where higher is better
                                overfitting_metrics[f"{metric}_overfit_ratio"] = test_mean / train_mean
                
                # Convert CV results to regular metrics dict
                metrics = {}
                
                # Process classification metrics
                if is_classification:
                    if len(np.unique(y)) == 2:  # Binary
                        metrics['accuracy'] = np.mean(cv_results['test_accuracy'])
                        metrics['roc_auc'] = np.mean(cv_results['test_roc_auc'])
                        metrics['f1'] = np.mean(cv_results['test_f1'])
                    else:  # Multi-class
                        metrics['accuracy'] = np.mean(cv_results['test_accuracy'])
                        metrics['f1_weighted'] = np.mean(cv_results['test_f1_weighted'])
                else:  # Regression
                    metrics['r2'] = np.mean(cv_results['test_r2'])
                    metrics['mean_absolute_error'] = -np.mean(cv_results['test_neg_mean_absolute_error'])
                    metrics['root_mean_squared_error'] = -np.mean(cv_results['test_neg_root_mean_squared_error'])
                
                # Add overfitting metrics
                metrics.update(overfitting_metrics)
                
                # Add feature count and memory usage
                metrics['n_features'] = len(X.columns)
                metrics['memory_usage'] = data.memory_usage().sum() / 1024**2  # MB
                
                return metrics
                
            except Exception as e:
                logger.error(f"Error in cross-validation: {str(e)}")
                
                # Try direct train-test split as fallback
                try:
                    from sklearn.model_selection import train_test_split
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42,
                        stratify=y if is_classification else None
                    )
                    
                    # Train model without early stopping
                    try:
                        # Get XGBoost version
                        xgb_version = xgb.__version__
                        major_version = 0
                        try:
                            major_version = int(xgb_version.split('.')[0])
                        except (ValueError, IndexError) as ve:
                            logger.warning(f"Error parsing XGBoost version: {str(ve)}")
                        
                        # First attempt using normal fit
                        if major_version >= 2:
                            # XGBoost 2.0+ approach
                            model.fit(X_train, y_train)
                        else:
                            # Pre-2.0 approach
                            model.fit(X_train, y_train)
                    except Exception as fit_error:
                        logger.error(f"Error in model fitting: {str(fit_error)}")
                        # Very simplified model
                        if is_classification:
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
                        else:
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
                        
                        model.fit(X_train, y_train)
                    
                    # Calculate metrics
                    metrics = {}
                    
                    # Basic feature information
                    metrics['n_features'] = len(X.columns)
                    metrics['memory_usage'] = data.memory_usage().sum() / 1024**2  # MB
                    
                    if is_classification:
                        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
                        
                        # Test predictions
                        y_pred = model.predict(X_test)
                        
                        metrics['accuracy'] = accuracy_score(y_test, y_pred)
                        
                        if len(np.unique(y)) == 2:  # Binary classification
                            try:
                                y_proba = model.predict_proba(X_test)[:, 1]
                                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                            except:
                                logger.warning("Could not calculate ROC AUC score - using accuracy")
                                metrics['roc_auc'] = metrics['accuracy']
                            
                            metrics['f1'] = f1_score(y_test, y_pred)
                        else:
                            metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
                        
                        # Train predictions for overfitting assessment
                        y_train_pred = model.predict(X_train)
                        train_acc = accuracy_score(y_train, y_train_pred)
                        
                        # Calculate overfitting ratio
                        if train_acc > 0:
                            metrics['accuracy_overfit_ratio'] = metrics['accuracy'] / train_acc
                    else:
                        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                        
                        # Test predictions
                        y_pred = model.predict(X_test)
                        
                        metrics['r2'] = r2_score(y_test, y_pred)
                        metrics['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
                        metrics['root_mean_squared_error'] = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        # Train predictions for overfitting assessment
                        y_train_pred = model.predict(X_train)
                        train_r2 = r2_score(y_train, y_train_pred)
                        train_mae = mean_absolute_error(y_train, y_train_pred)
                        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                        
                        # Calculate overfitting ratios (avoiding division by zero)
                        if train_r2 > 0:
                            metrics['r2_overfit_ratio'] = metrics['r2'] / train_r2
                        if train_mae > 0:
                            metrics['neg_mean_absolute_error_overfit_ratio'] = train_mae / metrics['mean_absolute_error']
                        if train_rmse > 0:
                            metrics['neg_root_mean_squared_error_overfit_ratio'] = train_rmse / metrics['root_mean_squared_error']
                    
                    return metrics
                
                except Exception as fallback_error:
                    logger.error(f"Fallback evaluation also failed: {str(fallback_error)}")
                    # Return minimal metrics
                    return {
                        'n_features': len(X.columns) if X is not None else 0,
                        'memory_usage': data.memory_usage().sum() / 1024**2 if data is not None else 0,
                        'error': str(e)
                    }
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                "error": str(e),
                "n_features": len(data.columns) - 1,
                "memory_usage": data.memory_usage().sum() / 1024**2,  # MB
            }
    
    def _save_results(self, result: PipelineResult) -> None:
        """Save final pipeline results."""
        if not self.config.output_dir:
            return
            
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save transformed data
        result.transformed_data.to_csv(
            output_dir / "final_data.csv",
            index=False,
        )
        
        # Save metadata
        with open(output_dir / "feature_metadata.json", "w") as f:
            json.dump(result.feature_metadata, f, indent=2)
        
        # Save performance metrics
        with open(output_dir / "performance_metrics.json", "w") as f:
            json.dump(result.performance_metrics, f, indent=2)
        
        # Save execution log
        with open(output_dir / "execution_log.json", "w") as f:
            json.dump(result.execution_log, f, indent=2) 