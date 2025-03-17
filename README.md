# FASTER: Feature Automation, Selection, Transformation, Extraction Routine

by: Carter Tran, Suchit Bhayani

FASTER is a Python framework designed to leverage Large Language Models (LLMs) for automating feature engineering in machine learning workflows. The system incorporates domain knowledge to generate, transform, and select optimal features, significantly reducing the manual effort typically required in feature engineering while improving model performance.

## Features

- Automated feature discovery using LLM-extracted domain knowledge
- Statistical evaluation and selection of significant features
- Comprehensive feature transformation pipeline
- Integration with OpenRouter API for LLM access
- Robust error handling and logging
- Detailed feature metadata and documentation
- Type-safe implementation with comprehensive test coverage
- Configurable pipeline with sensible defaults
- Extensive documentation and examples
- Interactive Streamlit web application for easy experimentation

## Requirements

- Python 3.9+
- [Rye](https://github.com/astral-sh/rye) (recommended) or pip
- OpenRouter API key for LLM access
- Streamlit (optional, for running the interactive web application)

## Recent Improvements

The FASTER framework has undergone several key improvements to address performance issues and enhance feature engineering capabilities:

### OpenRouter API Integration

- **Support for Multiple LLMs**: Full integration with OpenRouter API to access various LLM models
- **DeepSeek Integration**: Default configured to use `deepseek/deepseek-chat:free` model
- **Robust API Handling**: Improved error handling and logging for API interactions
- **Request Tracking**: Track all LLM interactions with unique request IDs
- **Response Validation**: Enhanced validation of LLM responses

### Anti-Overfitting Measures

- **XGBoost Integration**: Replaced RandomForest with highly regularized XGBoost models in feature selection and evaluation
- **Regularization Parameters**: Implemented L1/L2 regularization, controlled tree depth, and subsample ratios
- **Early Stopping**: Added early stopping during model training to prevent memorization of training data
- **Cross-Validation**: Expanded cross-validation to better estimate model performance
- **Overfitting Ratio Tracking**: New metrics track the ratio of test to train performance to identify overfitting
- **Stratified Sampling**: Ensures class distributions are maintained in validation splits
- **Shallow Decision Trees**: Using max_depth=3 to create simpler, more generalizable models
- **Feature Subsampling**: Using colsample_bytree parameter to evaluate features on different subsets

### Enhanced Feature Selection

- **Cross-Validation Feature Importance**: Now uses cross-validation to calculate more robust feature importance scores
- **Stability Selection**: Implements stability selection to identify consistently important features across bootstrapped samples
- **Variance Inflation Factor (VIF)**: Added checks for multicollinearity to reduce redundancy
- **Minimum Importance Threshold**: Features below a minimum importance score are automatically filtered out

### Smarter Feature Generation

- **Performance-Based Feature Filtering**: Generated features are evaluated for their utility and only beneficial transformations are kept
- **Transformation Prioritization**: Features are prioritized based on domain importance
- **Reduced Correlation Threshold**: Decreased from 0.9 to 0.8 to be more aggressive with correlated features
- **Limited Transformation Counts**: Controls the number of each transformation type to prevent feature explosion
- **Performance Gain Tracking**: Each transformation is tracked for its contribution to model performance

### Improved Statistical Evaluation

- **Enhanced Variable Type Handling**: Better detection and processing of categorical vs. continuous variables
- **Predictive Power Assessment**: Individual features are evaluated for their predictive power
- **Automatic Categorical Detection**: More sophisticated detection of categorical columns
- **Robust Statistical Testing**: Tests are selected based on data characteristics

### Domain Knowledge Integration

- **Selective Transformation Application**: More selective about which transformations to apply based on domain insights
- **Top Feature Focus**: Concentrates transformations on features identified as most important by domain knowledge
- **Explicit Transformation Recommendations**: Better utilization of specific transformation suggestions

### Testing & Reliability

- **Improved Test Coverage**: Enhanced test suite with more comprehensive coverage
- **Integration Tests**: Additional tests for end-to-end pipeline functionality
  - **Multiple Dataset Testing**: Tests now include Iris, Titanic, Auto MPG, and Horsepower-MPG datasets
  - **Mock and Live Testing**: Integration tests support both mock LLM responses and live OpenRouter API calls
- **Performance Benchmarks**: Tests now verify actual performance improvements
- **Model Validation**: Validation of model improvements across different datasets
- **Automatic Test Configuration**: Tests automatically detect and configure API keys when available

These improvements have significantly enhanced the framework's ability to generate features that actually improve model performance, while reducing noise from unhelpful transformations.

## Installation

### Using Rye (Recommended)

```bash
# Clone the repository
git clone https://github.com/CarterT27/FASTER.git
cd FASTER

# Install dependencies using Rye
rye sync

# Activate the virtual environment
. .venv/bin/activate

# Install Streamlit dependencies (optional)
rye sync --extras=streamlit
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/CarterT27/FASTER.git
cd FASTER

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -e .

# Install Streamlit dependencies (optional)
pip install -e ".[streamlit]"
```

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
YOUR_SITE_URL=https://github.com/cartert27/FASTER
YOUR_SITE_NAME=FASTER Feature Engineering

# Optional settings
LOG_LEVEL=INFO
CACHE_DIR=.cache
```

3. Obtain an OpenRouter API key:
   - Visit [OpenRouter](https://openrouter.ai/) and create an account
   - Generate an API key from your dashboard
   - Add the key to your `.env` file
   - Your API key will allow access to various LLM models, including the default `deepseek/deepseek-chat:free` model

### OpenRouter API Integration Details

The FASTER framework uses OpenRouter to access various LLM models for domain knowledge extraction. The integration provides:

- Configurable model selection via the `model_name` parameter in `PipelineConfig`
- Support for various LLM parameters like temperature and max_tokens
- Built-in error handling and retries for API failures
- Automatic masking of API keys in logs for security
- Request tracking with unique IDs for each LLM interaction
- Response validation to ensure well-formatted outputs

The API key can be provided in several ways:
1. Through the `.env` file (recommended)
2. As an environment variable
3. Directly to the Pipeline constructor via the `api_key` parameter
4. In the Streamlit interface for the web application

## Project Structure

```
FASTER/
├── faster/                    # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── domain_knowledge.py   # Domain knowledge extraction
│   ├── feature_generation.py # Feature generation and transformation
│   ├── statistical_evaluation.py  # Statistical testing and evaluation
│   ├── feature_selection.py  # Feature selection
│   ├── pipeline.py          # Main pipeline orchestration
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── logging.py
│       └── validation.py
├── app.py                    # Streamlit web application
├── tests/                    # Test directory
│   ├── conftest.py          # Test configuration and shared fixtures
│   ├── test_domain_knowledge.py  # Domain knowledge extractor tests
│   ├── test_feature_generation.py  # Feature generation tests
│   ├── test_pipeline.py     # Pipeline tests
│   ├── test_iris_integration.py  # Integration test with Iris dataset
│   ├── test_titanic_integration.py  # Integration test with Titanic dataset
│   ├── test_auto_mpg_integration.py  # Integration test with Auto MPG dataset
│   └── test_horsepower_mpg_integration.py  # Integration test with Horsepower-MPG dataset
├── data/                     # Data directory
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── examples/                 # Example notebooks and scripts
│   └── quickstart.ipynb     # Quickstart notebook (placeholder - see Quick Start section for code example)
├── .env                     # Environment variables
├── .gitignore              # Git ignore rules
├── pyproject.toml          # Project configuration
├── requirements.lock       # Locked dependencies
├── README-streamlit.md     # Streamlit application documentation
└── README.md              # This file
```

## Quick Start

The code example below demonstrates the basic usage of FASTER. For more comprehensive examples and interactive usage, please refer to the Streamlit application (`app.py`).

```python
from faster import Pipeline
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize pipeline with custom configuration
from faster.pipeline import PipelineConfig
from faster.feature_selection import SelectionCriteria

config = PipelineConfig(
    model_name="deepseek/deepseek-chat:free",  # Default LLM model via OpenRouter
    temperature=0.0,
    max_interaction_degree=2,
    alpha=0.05,
    selection_criteria=SelectionCriteria(
        p_value_threshold=0.05,
        min_effect_size=0.1,
        max_correlation=0.8,
        min_importance_score=0.02,
        vif_threshold=10.0,
        cv_folds=5,
        stability_threshold=0.7,
    ),
    output_dir="results",
    save_intermediate=True,
)

pipeline = Pipeline(config)

# Run feature engineering
result = pipeline.run(
    data=data,
    target_column="target",
    problem_description="Predict customer churn based on usage patterns",
    categorical_columns=["plan_type", "region"],
    domain_context="Telecommunications industry with monthly subscription model",
    is_classification=True,
)

# Access results
transformed_data = result.transformed_data
selected_features = result.selected_features
feature_metadata = result.feature_metadata

# See performance metrics and feature importance
print(f"Selected {len(selected_features)} features")
for feature, metadata in feature_metadata.items():
    print(f"{feature}: importance={metadata.get('importance_score', 0):.4f}")
    if 'transformation' in metadata and metadata['transformation']:
        print(f"  - Transformation: {metadata['transformation'].get('transformation_type', '')}")
        print(f"  - Performance gain: {metadata['transformation'].get('performance_gain', 'N/A')}")
```

This example demonstrates the basic workflow of using FASTER. For more detailed examples and interactive usage, use the Streamlit application.

## Feature Generation Types

FASTER implements a comprehensive set of feature transformations, automatically selecting and applying the most appropriate ones based on data characteristics and domain knowledge. Each transformation is tracked with metadata including original features, parameters, and rationale.

### Performance-Based Transformation Evaluation

A key improvement in FASTER is the performance-based evaluation of transformations:

- **Transformation Utility Assessment**: Each transformation is evaluated for its contribution to model performance
- **Baseline Comparison**: Transformed features are compared against a baseline model with original features
- **Selective Retention**: Only transformations that improve performance beyond a threshold are kept
- **Performance Gain Tracking**: Each transformation is annotated with its quantified performance contribution
- **Group Evaluation**: Transformations of similar types are evaluated together for efficiency

### Basic Numerical Transformations
- **Log Transform**: Applied to skewed features (`log_*`)
  - Automatic detection based on skewness > 1.0
  - Uses `np.log1p` for handling zero values
- **Standard Scaling**: Standardization of features (`scaled_*`)
  - Zero mean and unit variance
  - Tracked with individual scalers per feature

### Statistical Transformations
- **Z-Score**: Standardization using mean and standard deviation (`zscore_*`)
- **Min-Max Scaling**: Scale features to [0,1] range (`minmax_*`)
- **Box-Cox**: Power transformation for positive data (`boxcox_*`)
- **Yeo-Johnson**: Power transformation supporting negative values (`yeojohnson_*`)
- **Quantile**: Transform to normal distribution (`quantile_*`)
- **Binning**: Equal-frequency binning with customizable bins (`binned_*`)
  - Default: 5 bins
  - Supports custom bin counts

### Advanced Mathematical Transformations
- **Power**: Raise features to specified power (`power_*`)
  - Default power: 2
  - Configurable power parameter
- **Square**: Square transformation (`square_*`)
- **Cube**: Cubic transformation (`cube_*`)
- **Square Root**: For non-negative features (`sqrt_*`)
- **Logit**: Transform probabilities to log-odds (`logit_*`)
  - Handles values in [0,1] range
  - Uses epsilon adjustment for numerical stability
- **Sigmoid**: Logistic function transformation (`sigmoid_*`)

### Signal Processing Transformations
- **Detrend**: Remove linear trends (`detrend_*`)
- **Savitzky-Golay**: Smoothing filter (`savgol_*`)
  - Configurable window size
  - Polynomial order: 3
- **Hilbert**: Complex envelope detection (`hilbert_*`)
  - Extracts amplitude envelope

### Interaction Features
- **Multiplicative**: Pairwise multiplication (`multiply_*`)
  - Based on domain knowledge relationships
  - Automatic generation for related features
- **Differences**: Feature subtraction (`diff_*`)
  - Generated between related features
- **Ratios**: Feature division (`ratio_*`)
  - Includes zero-division protection

### Text Features
- **TF-IDF**: Term frequency-inverse document frequency (`tfidf_*`)
  - Maximum 10 features per text column
  - Automatic text column detection
  - Handles missing values
- **Text Detection**: Intelligent text column identification
  - Minimum 3 words per entry
  - Sampling for efficiency

### Domain-Specific Features
- Generated based on `DomainInsight` suggestions
- Custom transformations based on domain expertise
- Tracked with transformation metadata including rationale

Each transformation includes:
- Comprehensive error handling and logging
- Transformation metadata tracking
- Input validation and preprocessing
- Automatic feature naming
- Performance optimization

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
rye sync --dev

# Install pre-commit hooks
pre-commit install
```

### Code Style and Quality

We use the following tools to maintain code quality:

- **Ruff**: For fast Python linting
- **Black**: For code formatting
- **MyPy**: For static type checking
- **Pre-commit**: For automated checks before commits

```bash
# Run linting
ruff check .

# Run type checking
mypy .

# Run formatters
black .
```

### Testing

We use pytest for testing. Tests are located in the `tests/` directory and mirror the main package structure.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_feature_generation.py

# Run with coverage report
pytest --cov=faster
```

The test suite includes both unit tests for individual components and integration tests that verify the end-to-end pipeline with real datasets:
- **Iris Classification**: Tests classification performance on the Iris flower dataset (`test_iris_integration.py`)
- **Titanic Classification**: Tests binary classification on the Titanic survival dataset (`test_titanic_integration.py`)
- **Auto MPG Regression**: Tests regression performance on the Auto MPG dataset (`test_auto_mpg_integration.py`)
- **Horsepower-MPG Regression**: Tests regression on the Horsepower vs. MPG relationship (`test_horsepower_mpg_integration.py`)

These integration tests ensure that our feature engineering actually improves model performance in typical scenarios. Tests run both with mocked LLM responses (for fast, deterministic testing) and with live API calls (for complete end-to-end validation).

### Testing OpenRouter API

The repository includes a simple test script `test_openrouter.py` that can be used to verify your OpenRouter API key is working correctly:

```bash
# Set your API key in the environment
export OPENROUTER_API_KEY="your-api-key-here"

# Run the test script
python test_openrouter.py
```

This will attempt a simple completion request to verify API connectivity.

Tests have been updated to accommodate recent improvements in the framework, particularly around:

1. The more selective approach to feature transformation
2. Performance-based feature evaluation
3. Cross-validation for feature importance
4. Stability selection for consistent feature importance

This ensures that the tests validate the framework's ability to:
- Generate useful transformations while filtering out noise
- Apply domain knowledge appropriately
- Handle both classification and regression tasks
- Work with small and medium-sized datasets efficiently

### Logging

FASTER uses Python's built-in logging module with enhanced formatting:

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about pipeline progress
- **WARNING**: Warnings about potential issues
- **ERROR**: Errors that don't halt execution
- **CRITICAL**: Critical errors that stop execution

Configure logging level in `.env` or programmatically:

```python
import logging
logging.getLogger("faster").setLevel(logging.DEBUG)
```

## Error Handling

FASTER provides custom exceptions for different error categories:

- `FasterValidationError`: Input validation errors
- `FasterConfigError`: Configuration errors
- `FasterAPIError`: External API communication errors
  - Includes specialized handling for OpenRouter API errors
  - Retry logic for transient API issues
  - Rate limiting detection and backoff
  - Automatic fallback to alternative models when possible
- `FasterTransformError`: Feature transformation errors
- `FasterStatisticalError`: Statistical evaluation errors

All exceptions include detailed context information for debugging, including:
- Timestamp of the error
- Request ID for API calls
- Stack trace information
- Relevant input data (sanitized of sensitive information)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Create a Pull Request

### Pull Request Guidelines

- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Keep changes focused and atomic

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenRouter for LLM API access
- The scikit-learn community for statistical tools
- The Python data science community

## Interactive Web Application

FASTER includes an interactive Streamlit web application that allows you to:

1. Select from predefined datasets (Iris, Auto MPG, Titanic, Horsepower-MPG)
2. Upload your own datasets
3. Configure and run the FASTER pipeline
4. Compare performance metrics between baseline and FASTER models
5. Visualize feature importance and transformations

To run the web application:

```bash
streamlit run app.py
```

For detailed information about the Streamlit application, see the [README-streamlit.md](README-streamlit.md) file.
