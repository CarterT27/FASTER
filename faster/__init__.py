"""FASTER: Feature Automation, Selection, Transformation, Extraction Routine.

A Python framework for automated feature engineering using LLMs and statistical analysis.
"""

__version__ = "0.1.0"

from faster.pipeline import Pipeline
from faster.domain_knowledge import DomainKnowledgeExtractor
from faster.feature_generation import FeatureGenerator
from faster.statistical_evaluation import StatisticalEvaluator
from faster.feature_selection import FeatureSelector

__all__ = [
    "Pipeline",
    "DomainKnowledgeExtractor",
    "FeatureGenerator",
    "StatisticalEvaluator",
    "FeatureSelector",
]
