"""Streamlit application for demonstrating the FASTER Pipeline.

This app allows users to:
1. Select from predefined datasets (Iris, Auto MPG, Titanic, Horsepower-MPG)
2. Upload their own datasets
3. Configure and run the FASTER pipeline
4. Compare performance metrics between baseline and FASTER models
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import copy

from faster.pipeline import Pipeline, PipelineConfig
from faster.feature_selection import SelectionCriteria
from faster.domain_knowledge import DomainKnowledgeExtractor

from utils.datasets import (
    load_iris_dataset,
    load_auto_mpg_dataset,
    load_titanic_dataset,
    load_horsepower_mpg_dataset,
)
from utils.evaluation import evaluate_model
from utils.visualization import plot_feature_importance, plot_metrics_comparison

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FASTER Pipeline Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 FASTER Pipeline Demo")
st.markdown("""
This application demonstrates the **Feature Automation, Selection, Transformation, Extraction Routine (FASTER)** 
pipeline for automated feature engineering. The pipeline utilizes domain knowledge from large language models
to enhance model performance.

You can either select from predefined datasets or upload your own dataset to see how FASTER improves model performance.
""")


def main():
    """Main function to run the Streamlit app."""

    st.sidebar.title("Configuration")

    dataset_option = st.sidebar.radio(
        "Select Dataset Source", ["Predefined Dataset", "Upload Dataset"]
    )

    if dataset_option == "Predefined Dataset":
        data, target_column, is_classification, problem_description = load_predefined_dataset()
    else:
        data, target_column, is_classification, problem_description = load_custom_dataset()

    if data is not None:
        with st.expander("Dataset Information", expanded=True):
            st.write(f"Dataset shape: {data.shape}")
            st.write(f"Target column: {target_column}")
            st.write(f"Problem type: {'Classification' if is_classification else 'Regression'}")
            st.dataframe(data.head())

        st.sidebar.header("Pipeline Configuration")

        st.sidebar.subheader("LLM Configuration")
        api_key = st.sidebar.text_input("OpenRouter API Key (optional)", type="password")

        use_mock = not bool(api_key.strip())
        if use_mock:
            st.sidebar.warning("No API key provided. Using mock LLM responses.")

        model_name = st.sidebar.selectbox(
            "LLM Model",
            [
                "deepseek/deepseek-chat:free",
                "anthropic/claude-3-sonnet:free",
                "mistral/mistral-large:free",
            ],
            index=0,
        )

        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

        st.sidebar.subheader("Feature Engineering Configuration")
        max_interaction_degree = st.sidebar.slider("Max Interaction Degree", 1, 3, 2)
        alpha = st.sidebar.slider("Statistical Significance Level (alpha)", 0.01, 0.1, 0.05, 0.01)
        correction_method = st.sidebar.selectbox(
            "Multiple Testing Correction Method",
            ["fdr_bh", "bonferroni", "sidak", "holm", "none"],
            index=0,
        )

        if st.button("Run FASTER Pipeline"):
            with st.spinner("Running FASTER Pipeline..."):
                config = PipelineConfig(
                    model_name=model_name,
                    temperature=temperature,
                    max_interaction_degree=max_interaction_degree,
                    text_feature_method="tfidf",
                    alpha=alpha,
                    correction_method=correction_method,
                    selection_criteria=SelectionCriteria(),
                    output_dir=None,
                    save_intermediate=False,
                    keep_all_features=True,
                )

                results = run_pipeline(
                    data=data,
                    target_column=target_column,
                    problem_description=problem_description,
                    is_classification=is_classification,
                    config=config,
                    use_mock=use_mock,
                    api_key=api_key,
                )

                display_results(results, data, target_column, is_classification)


def load_predefined_dataset() -> Tuple[Optional[pd.DataFrame], Optional[str], bool, str]:
    """Load a predefined dataset based on user selection."""
    dataset_name = st.sidebar.selectbox(
        "Select Dataset", ["Iris", "Auto MPG", "Titanic", "Horsepower-MPG"], index=0
    )

    if dataset_name == "Iris":
        data, target = load_iris_dataset()
        is_classification = False  # Regression problem (predicting sepal length)
        problem_description = """
        Regression problem predicting sepal length of iris flowers.
        Features:
        - sepal_width: Width of the sepal in cm
        - petal_length: Length of the petal in cm
        - petal_width: Width of the petal in cm
        - species: Type of iris (setosa, versicolor, virginica)
        
        Domain context:
        - Different iris species have characteristic sepal and petal dimensions
        - Petal dimensions tend to be correlated with sepal dimensions
        - Setosa species has the smallest petals but relatively wide sepals
        - Virginica species has the largest petals and sepals
        - There are allometric relationships between different flower parts
        """
        return data, "sepal_length", is_classification, problem_description

    elif dataset_name == "Auto MPG":
        data, target = load_auto_mpg_dataset()
        is_classification = False  # Regression problem (predicting MPG)
        problem_description = """
        Regression problem predicting miles per gallon (MPG) of automobiles.
        Features:
        - cylinders: Number of cylinders in the engine (integer)
        - displacement: Engine displacement in cubic inches (continuous)
        - horsepower: Engine horsepower (continuous)
        - weight: Vehicle weight in pounds (continuous)
        - acceleration: Time to accelerate from 0 to 60 mph in seconds (continuous)
        - model_year: Vehicle model year (integer, 70-82 representing 1970-1982)
        - origin: Origin of car (categorical: american, european, asian)
        
        Domain context:
        - Fuel efficiency (MPG) generally decreases with vehicle weight
        - Higher horsepower engines typically consume more fuel
        - Larger engine displacement generally correlates with lower MPG
        - Cars with more cylinders typically have lower fuel efficiency
        - Technological improvements over model years generally increased fuel efficiency
        - There are tradeoffs between performance (acceleration) and fuel efficiency
        - Different regions (origins) had different emission standards and design philosophies
        - The dataset contains cars from the 1970s and early 1980s during fuel crises
        """
        return data, "mpg", is_classification, problem_description

    elif dataset_name == "Titanic":
        data, target = load_titanic_dataset()
        is_classification = True  # Classification problem (predicting survival)
        problem_description = """
        Binary classification problem predicting passenger survival on the Titanic.
        Features:
        - Pclass: Ticket class (1st, 2nd, 3rd)
        - Sex: Passenger's sex
        - Age: Passenger's age
        - SibSp: Number of siblings/spouses aboard
        - Parch: Number of parents/children aboard
        - Fare: Passenger fare
        - Embarked: Port of embarkation
        
        Domain context:
        - First class passengers had higher survival rates
        - Women and children had priority in lifeboats
        - Families traveling together might have different survival patterns
        - Ticket fare could indicate wealth and access to better areas of the ship
        - Port of embarkation could indicate passenger's social status
        """
        return data, "Survived", is_classification, problem_description

    elif dataset_name == "Horsepower-MPG":
        data, target = load_horsepower_mpg_dataset()
        is_classification = False  # Regression problem (predicting MPG)
        problem_description = """
        Regression problem predicting miles per gallon (MPG) of automobiles based only on horsepower.
        
        Domain context:
        - Higher horsepower engines typically consume more fuel, resulting in lower MPG
        - The relationship between horsepower and MPG is generally non-linear
        - There may be diminishing returns where increases in horsepower above certain thresholds have less impact on MPG
        """
        return data, "mpg", is_classification, problem_description

    return None, None, False, ""


def load_custom_dataset() -> Tuple[Optional[pd.DataFrame], Optional[str], bool, str]:
    """Load a custom dataset from user upload."""
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            target_column = st.sidebar.selectbox("Select Target Column", data.columns.tolist())

            problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"])
            is_classification = problem_type == "Classification"

            problem_description = st.sidebar.text_area(
                "Problem Description (include domain knowledge if available)",
                height=150,
                placeholder="Describe the problem and any domain knowledge you have...",
            )

            return data, target_column, is_classification, problem_description

        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

    return None, None, False, ""


def run_pipeline(
    data: pd.DataFrame,
    target_column: str,
    problem_description: str,
    is_classification: bool,
    config: PipelineConfig,
    use_mock: bool,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the FASTER pipeline and return results.

    Args:
        data: Input dataframe with target column
        target_column: Name of the target column
        problem_description: Description of the problem and domain knowledge
        is_classification: Whether it's a classification problem
        config: Pipeline configuration
        use_mock: Whether to use mock LLM responses
        api_key: Optional OpenRouter API key to pass to LLM-based components

    Returns:
        Dictionary containing pipeline results and evaluation metrics
    """

    data_copy = data.copy()

    X = data_copy.drop(target_column, axis=1)
    y = data_copy[target_column]

    st.info("Evaluating baseline model...")
    baseline_scores = evaluate_model(X, y, is_classification)

    pipeline_no_domain = Pipeline(config)

    domain_config = copy.deepcopy(config)
    pipeline_with_domain = Pipeline(domain_config, api_key=api_key)

    st.info("Running FASTER pipeline without domain knowledge...")
    result_no_domain = pipeline_no_domain.run(
        data=data_copy,
        target_column=target_column,
        problem_description="Predicting " + target_column,
        is_classification=is_classification,
        keep_all_features=True,
    )

    st.info("Running FASTER pipeline with domain knowledge...")
    result_with_domain = pipeline_with_domain.run(
        data=data_copy,
        target_column=target_column,
        problem_description=problem_description,
        is_classification=is_classification,
        keep_all_features=True,
    )

    no_domain_features = result_no_domain.transformed_data
    with_domain_features = result_with_domain.transformed_data

    if target_column in no_domain_features.columns:
        no_domain_features = no_domain_features.drop(target_column, axis=1)
    if target_column in with_domain_features.columns:
        with_domain_features = with_domain_features.drop(target_column, axis=1)

    st.info("Evaluating FASTER models...")
    faster_no_domain_scores = evaluate_model(no_domain_features, y, is_classification)
    faster_with_domain_scores = evaluate_model(with_domain_features, y, is_classification)

    return {
        "baseline_scores": baseline_scores,
        "faster_no_domain_scores": faster_no_domain_scores,
        "faster_with_domain_scores": faster_with_domain_scores,
        "no_domain_features": no_domain_features,
        "with_domain_features": with_domain_features,
        "result_no_domain": result_no_domain,
        "result_with_domain": result_with_domain,
    }


def display_results(
    results: Dict[str, Any], data: pd.DataFrame, target_column: str, is_classification: bool
):
    """
    Display pipeline results and evaluation metrics.

    Args:
        results: Pipeline results and evaluation metrics
        data: Original dataframe
        target_column: Name of the target column
        is_classification: Whether it's a classification problem
    """
    st.header("🔍 Results")

    baseline_scores = results["baseline_scores"]
    faster_no_domain_scores = results["faster_no_domain_scores"]
    faster_with_domain_scores = results["faster_with_domain_scores"]
    no_domain_features = results["no_domain_features"]
    with_domain_features = results["with_domain_features"]
    result_no_domain = results["result_no_domain"]
    result_with_domain = results["result_with_domain"]

    tabs = st.tabs(
        ["Performance Metrics", "Feature Analysis", "Data Transformation", "Pipeline Details"]
    )

    with tabs[0]:
        st.subheader("Model Performance Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Baseline Model")
            display_metrics(baseline_scores, is_classification, None)

        with col2:
            st.markdown("### FASTER (No Domain Knowledge)")
            display_metrics(faster_no_domain_scores, is_classification, baseline_scores)

        with col3:
            st.markdown("### FASTER (With Domain Knowledge)")
            display_metrics(faster_with_domain_scores, is_classification, baseline_scores)

        st.subheader("Metrics Comparison")
        fig = plot_metrics_comparison(
            baseline_scores, faster_no_domain_scores, faster_with_domain_scores, is_classification
        )
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Feature Analysis")

        st.markdown("### Feature Counts")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Original Features", len(data.drop(target_column, axis=1).columns))

        with col2:
            st.metric(
                "FASTER (No Domain) Features",
                len(no_domain_features.columns),
                len(no_domain_features.columns) - len(data.drop(target_column, axis=1).columns),
            )

        with col3:
            st.metric(
                "FASTER (With Domain) Features",
                len(with_domain_features.columns),
                len(with_domain_features.columns) - len(data.drop(target_column, axis=1).columns),
            )

        st.markdown("### Selected Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### FASTER (No Domain Knowledge)")
            if (
                hasattr(result_no_domain, "selected_features")
                and result_no_domain.selected_features
            ):
                st.write(sorted(result_no_domain.selected_features))
            else:
                st.write("No features selected")

        with col2:
            st.markdown("#### FASTER (With Domain Knowledge)")
            if (
                hasattr(result_with_domain, "selected_features")
                and result_with_domain.selected_features
            ):
                st.write(sorted(result_with_domain.selected_features))
            else:
                st.write("No features selected")

        st.markdown("### Feature Importance")
        try:
            if (
                hasattr(result_with_domain, "feature_importances")
                and result_with_domain.feature_importances is not None
            ):
                fig = plot_feature_importance(result_with_domain.feature_importances)
                st.pyplot(fig)
            else:
                st.info("Feature importance information not available")
        except Exception as e:
            st.error(f"Could not plot feature importance: {str(e)}")

    with tabs[2]:
        st.subheader("Data Transformation")

        st.markdown("### Transformed Data Samples")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### FASTER (No Domain Knowledge)")
            st.dataframe(no_domain_features.head())

        with col2:
            st.markdown("#### FASTER (With Domain Knowledge)")
            st.dataframe(with_domain_features.head())

    with tabs[3]:
        st.subheader("Pipeline Details")

        st.markdown("### Domain Knowledge Extraction")

        if hasattr(result_with_domain, "domain_knowledge") and result_with_domain.domain_knowledge:
            st.json(result_with_domain.domain_knowledge)
        else:
            st.info("Domain knowledge information not available")

        st.markdown("### Statistical Tests")

        if (
            hasattr(result_with_domain, "statistical_tests")
            and result_with_domain.statistical_tests
        ):
            st.json(result_with_domain.statistical_tests)
        else:
            st.info("Statistical tests information not available")


def display_metrics(
    metrics: Dict[str, float], is_classification: bool, baseline_metrics: Optional[Dict[str, float]]
):
    """
    Display model performance metrics.

    Args:
        metrics: Dictionary of metrics
        is_classification: Whether it's a classification problem
        baseline_metrics: Baseline metrics for comparison (optional)
    """

    test_metrics = {k.replace("test_", ""): v for k, v in metrics.items() if k.startswith("test_")}

    train_metrics = {
        k.replace("train_", ""): v for k, v in metrics.items() if k.startswith("train_")
    }

    st.markdown("#### Test Metrics")
    for metric, value in test_metrics.items():
        if baseline_metrics is not None:
            baseline_value = baseline_metrics.get(f"test_{metric}", 0)
            delta = value - baseline_value

            if metric in ["mae", "rmse"]:
                delta = -delta

            st.metric(label=metric, value=f"{value:.4f}", delta=f"{delta:.4f}")
        else:
            st.metric(label=metric, value=f"{value:.4f}")

    st.markdown("#### Train Metrics")
    for metric, value in train_metrics.items():
        if baseline_metrics is not None:
            baseline_value = baseline_metrics.get(f"train_{metric}", 0)
            delta = value - baseline_value

            if metric in ["mae", "rmse"]:
                delta = -delta

            st.metric(label=metric, value=f"{value:.4f}", delta=f"{delta:.4f}")
        else:
            st.metric(label=metric, value=f"{value:.4f}")


if __name__ == "__main__":
    main()
