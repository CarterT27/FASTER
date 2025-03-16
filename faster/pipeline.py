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
        
        self.feature_selector = FeatureSelector(
            criteria=self.config.selection_criteria,
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
    ) -> PipelineResult:
        """Run the full FASTER pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            problem_description: Description of the problem to solve
            categorical_columns: List of categorical column names
            domain_context: Optional additional domain context
            is_classification: Whether this is a classification task
            
        Returns:
            Results of the pipeline including transformed data and metadata
        """
        # Validate inputs
        if data is None:
            raise ValueError("Data cannot be None")
            
        if not target_column:
            raise ValueError("Target column cannot be empty")
            
        if not problem_description:
            raise ValueError("Problem description cannot be empty")
            
        # Further validate the dataframe
        validate_dataframe(data, target_column, categorical_columns=categorical_columns)
        
        start_time = time.time()
        run_id = str(uuid.uuid4())
        logger.info("Starting FASTER pipeline")
        
        execution_log = {
            "run_id": run_id,
            "start_time": datetime.datetime.now().isoformat(),
            "config": self.config.dict() if hasattr(self.config, "dict") else vars(self.config),
            "steps": {}
        }
        
        try:
            # 1. Extract domain knowledge
            logger.info("Extracting domain knowledge")
            step_start = time.time()
            
            if domain_context is None:
                domain_context = ""
            
            # Create dataset summary for LLM context
            dataset_summary = self._create_dataset_summary(data, target_column, categorical_columns)
            
            # Extract domain insights
            domain_insights = self.domain_extractor.extract_knowledge(
                data=data,
                target_column=target_column,
                problem_description=problem_description,
                request_id=run_id
            )
            
            execution_log["steps"]["domain_knowledge"] = {
                "duration": time.time() - step_start,
                "num_insights": len(domain_insights),
            }
            
            # 2. Generate features based on domain knowledge
            logger.info("Generating features")
            step_start = time.time()
            
            # Keep a copy of original data for performance comparison
            original_data = data.copy()
            
            transformed_data = self.feature_generator.generate_features(
                data=data,
                domain_insights=domain_insights,
                target_column=target_column,
                is_classification=is_classification,
            )
            
            execution_log["steps"]["feature_generation"] = {
                "duration": time.time() - step_start,
                "original_features": len(data.columns) - 1,  # exclude target
                "generated_features": len(transformed_data.columns) - len(data.columns),
                "total_features": len(transformed_data.columns) - 1,  # exclude target
            }
            
            # 3. Evaluate features statistically
            logger.info("Evaluating features")
            step_start = time.time()
            
            try:
                feature_stats = self.statistical_evaluator.evaluate_features(
                    data=transformed_data,
                    target_column=target_column,
                    categorical_columns=categorical_columns,
                )
            except Exception as e:
                logger.error(f"Error in statistical evaluation: {str(e)}")
                # Create fallback feature stats
                feature_stats = self._create_fallback_feature_stats(transformed_data, target_column)
            
            execution_log["steps"]["statistical_evaluation"] = {
                "duration": time.time() - step_start,
                "num_stats": len(feature_stats),
            }
            
            # 4. Select optimal feature subset
            logger.info("Selecting features")
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
                }
            
            # 5. Calculate performance metrics
            step_start = time.time()
            try:
                # Calculate performance with transformed features
                transformed_metrics = self._calculate_performance_metrics(
                    data=final_data,
                    target_column=target_column,
                    is_classification=is_classification,
                )
                
                # Calculate baseline performance with original features
                baseline_metrics = self._calculate_performance_metrics(
                    data=original_data,
                    target_column=target_column,
                    is_classification=is_classification,
                )
                
                # Compare performance metrics to determine if we should use transformed features
                use_transformed_features = self._compare_metrics(transformed_metrics, baseline_metrics, is_classification)
                
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
                        logger.error(f"Error reverting to baseline: {str(e)}")
                        # Keep what we have but with baseline metrics
                        performance_metrics = baseline_metrics
                else:
                    # Use the transformed features
                    performance_metrics = transformed_metrics
                
                # Add comparison to execution log
                execution_log["steps"]["performance_comparison"] = {
                    "baseline_metrics": baseline_metrics,
                    "transformed_metrics": transformed_metrics,
                    "used_transformed_features": use_transformed_features,
                }
                
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {str(e)}")
                performance_metrics = {"error": str(e)}
            
            execution_log["steps"]["performance_evaluation"] = {
                "duration": time.time() - step_start,
                "metrics": performance_metrics,
            }
            
            # 6. Save results if output directory is specified
            if self.config.output_dir:
                self._save_results(PipelineResult(
                    transformed_data=final_data,
                    selected_features=selection_result.selected_features,
                    feature_metadata=feature_metadata,
                    performance_metrics=performance_metrics,
                    execution_log=execution_log,
                ))
            
            # Update execution log with total runtime
            execution_log["total_duration"] = time.time() - start_time
            execution_log["end_time"] = datetime.datetime.now().isoformat()
            
            return PipelineResult(
                transformed_data=final_data,
                selected_features=selection_result.selected_features,
                feature_metadata=feature_metadata,
                performance_metrics=performance_metrics,
                execution_log=execution_log,
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            # Return partial results if possible
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