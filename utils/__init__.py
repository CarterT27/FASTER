"""Utilities for FASTER Streamlit App."""

from utils.datasets import (
    load_iris_dataset,
    load_auto_mpg_dataset,
    load_titanic_dataset,
    load_horsepower_mpg_dataset
)
from utils.evaluation import evaluate_model
from utils.visualization import plot_metrics_comparison, plot_feature_importance 