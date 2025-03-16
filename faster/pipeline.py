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

from faster.domain_knowledge import DomainKnowledgeExtractor
from faster.feature_generation import FeatureGenerator, TransformationMetadata
from faster.statistical_evaluation import StatisticalEvaluator, FeatureStatistics
from faster.feature_selection import FeatureSelector, SelectionCriteria, SelectionResult
from faster.utils.logging import get_logger, setup_logging

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
        """
        Run the FASTER pipeline.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            problem_description (str): Description of the problem
            categorical_columns (List[str], optional): List of categorical columns
            domain_context (str, optional): Additional domain context
            is_classification (bool): Whether this is a classification problem
            
        Returns:
            PipelineResult: Results from the pipeline
        """
        try:
            start_time = time.time()
            logger.info("Starting FASTER pipeline")
            
            # Validate inputs
            if data is None:
                logger.error("Data cannot be None")
                raise ValueError("Data cannot be None")
                
            if not target_column:
                logger.error("Target column cannot be empty")
                raise ValueError("Target column cannot be empty")
                
            if not problem_description:
                logger.error("Problem description cannot be empty")
                raise ValueError("Problem description cannot be empty")
            
            # Create a copy of the data to avoid modifying the original
            data = data.copy()
            
            # If domain context is not provided, use problem_description
            if domain_context is None:
                domain_context = problem_description
            
            # Track execution logs
            execution_log = {}
            
            # Step 1: Extract domain knowledge
            logger.info("Extracting domain knowledge")
            domain_insights = []
            domain_error = None
            
            try:
                # Use the request ID for tracking
                request_id = str(uuid.uuid4())
                domain_insights = self.domain_extractor.extract_knowledge(
                    data=data,
                    target_column=target_column,
                    problem_description=domain_context,
                    request_id=request_id,
                )
                execution_log["domain_extraction"] = {
                    "status": "success",
                    "request_id": request_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            except Exception as e:
                # Log the error but continue with an empty domain insights list
                domain_error = str(e)
                logger.error(f"Error in domain knowledge extraction: {domain_error}")
                domain_insights = []
                execution_log["domain_extraction"] = {
                    "status": "error",
                    "error": domain_error,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            
            if self.config.save_intermediate:
                self._save_intermediate("domain_insights.json", [insight.dict() for insight in domain_insights])
            
            # Step 2: Generate features
            logger.info("Generating features")
            transformed_data = data.copy()
            generation_error = None
            
            try:
                transformed_data = self.feature_generator.generate_features(
                    data=data,
                    domain_insights=domain_insights,
                    target_column=target_column,
                    is_classification=is_classification,
                )
                execution_log["feature_generation"] = {
                    "status": "success",
                    "features_added": len(transformed_data.columns) - len(data.columns),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            except Exception as e:
                generation_error = str(e)
                logger.error(f"Error in feature generation: {generation_error}")
                # Fall back to original data if feature generation fails
                transformed_data = data.copy()
                execution_log["feature_generation"] = {
                    "status": "error",
                    "error": generation_error,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            
            if self.config.save_intermediate:
                self._save_intermediate("transformed_data.csv", transformed_data)
            
            # Step 3: Evaluate features
            logger.info("Evaluating features")
            feature_stats = []
            evaluation_error = None
            
            try:
                feature_stats = self.statistical_evaluator.evaluate_features(
                    transformed_data,
                    target_column,
                    categorical_columns,
                )
                execution_log["feature_evaluation"] = {
                    "status": "success",
                    "features_evaluated": len(feature_stats),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            except Exception as e:
                evaluation_error = str(e)
                logger.error(f"Error in feature evaluation: {evaluation_error}")
                # Create minimal feature stats if evaluation fails
                feature_stats = self._create_fallback_feature_stats(transformed_data, target_column)
                execution_log["feature_evaluation"] = {
                    "status": "error",
                    "error": evaluation_error,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            
            if self.config.save_intermediate:
                self._save_intermediate("feature_stats.json", feature_stats)
            
            # Step 4: Select features
            logger.info("Selecting features")
            selection_result = None
            selection_error = None
            
            try:
                selection_result = self.feature_selector.select_features(
                    transformed_data,
                    target_column,
                    feature_stats,
                    is_classification,
                )
                execution_log["feature_selection"] = {
                    "status": "success",
                    "features_selected": len(selection_result.selected_features),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            except Exception as e:
                selection_error = str(e)
                logger.error(f"Error in feature selection: {selection_error}")
                # Create a fallback selection result with all columns except target
                all_features = [col for col in transformed_data.columns if col != target_column]
                selection_result = SelectionResult(
                    selected_features=all_features,
                    selection_scores={},
                    removed_features={},
                    statistics={f: FeatureStatistics(name=f) for f in all_features}
                )
                execution_log["feature_selection"] = {
                    "status": "error",
                    "error": selection_error,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            
            # Prepare results
            final_data = transformed_data[
                [target_column] + selection_result.selected_features
            ]
            
            feature_metadata = self._compile_feature_metadata(
                selection_result,
                self.feature_generator.transformations,
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                final_data,
                target_column,
                is_classification,
            )
            
            # Add pipeline execution time
            execution_time = time.time() - start_time
            execution_log["total_execution_time"] = execution_time
            
            # Create pipeline result
            result = PipelineResult(
                transformed_data=final_data,
                selected_features=selection_result.selected_features,
                feature_metadata=feature_metadata,
                performance_metrics=performance_metrics,
                execution_log=execution_log,
            )
            
            # Save results if output_dir is specified
            if self.config.output_dir:
                self._save_results(result)
            
            return result
            
        except Exception as e:
            # Catch any uncaught exceptions in the pipeline
            logger.error(f"Pipeline failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_fallback_feature_stats(self, data: pd.DataFrame, target_column: str) -> List[FeatureStatistics]:
        """Create minimal feature statistics when evaluation fails."""
        features = [col for col in data.columns if col != target_column]
        stats = []
        
        for feature in features:
            # Create a minimal FeatureStatistics object
            stat = FeatureStatistics(
                name=feature,
                p_value=0.5,  # Neutral p-value
                effect_size=0.0,
                predictive_power=0.0,
                is_categorical=not pd.api.types.is_numeric_dtype(data[feature])
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
        """Calculate performance metrics for the selected feature set."""
        # This would typically involve cross-validation with a simple model
        # Implementation depends on specific requirements
        return {
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