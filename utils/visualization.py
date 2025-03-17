"""Visualization utilities for FASTER Streamlit app."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union


def plot_metrics_comparison(
    baseline_scores: Dict[str, float],
    faster_no_domain_scores: Dict[str, float],
    faster_with_domain_scores: Dict[str, float],
    is_classification: bool
) -> plt.Figure:
    """
    Plot comparison of metrics across different models.
    
    Args:
        baseline_scores: Dictionary of baseline model scores
        faster_no_domain_scores: Dictionary of FASTER (no domain) model scores
        faster_with_domain_scores: Dictionary of FASTER (with domain) model scores
        is_classification: Whether it's a classification problem
        
    Returns:
        Matplotlib Figure object
    """

    if is_classification:
        metrics = ['accuracy', 'f1', 'roc_auc']
    else:
        metrics = ['r2', 'mae', 'rmse']

    plot_data = []
    
    for metric in metrics:
        baseline_value = baseline_scores.get(f'test_{metric}', 0)
        no_domain_value = faster_no_domain_scores.get(f'test_{metric}', 0)
        with_domain_value = faster_with_domain_scores.get(f'test_{metric}', 0)
        
        plot_data.append({
            'Metric': metric.upper(),
            'Model': 'Baseline',
            'Value': baseline_value
        })
        plot_data.append({
            'Metric': metric.upper(),
            'Model': 'FASTER (No Domain)',
            'Value': no_domain_value
        })
        plot_data.append({
            'Metric': metric.upper(),
            'Model': 'FASTER (With Domain)',
            'Value': with_domain_value
        })
    
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)

    ax.set_title('Model Performance Comparison (Test Metrics)', fontsize=14)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)

    ax.legend(title='Model')

    for metric in metrics:
        if metric in ['mae', 'rmse']:

            current_min, current_max = ax.get_ylim()
            if current_min > 0:
                ax.set_ylim(0, current_max * 1.1)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    feature_importances: Union[Dict[str, float], pd.Series],
    top_n: int = 15
) -> plt.Figure:
    """
    Plot feature importance from the model.
    
    Args:
        feature_importances: Dictionary or Series of feature importances
        top_n: Number of top features to show
        
    Returns:
        Matplotlib Figure object
    """

    if isinstance(feature_importances, dict):
        importances = pd.Series(feature_importances)
    else:
        importances = feature_importances

    importances = importances.sort_values(ascending=False)
    if len(importances) > top_n:
        importances = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, len(importances) * 0.3)))

    importances.plot(kind='barh', ax=ax)

    ax.set_title(f'Top {len(importances)} Feature Importances', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    return fig 