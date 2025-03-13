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

## Installation

```bash
# Using rye (recommended)
rye sync

# Using pip
pip install -e .
```

## Quick Start

```python
from faster import Pipeline
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize pipeline
pipeline = Pipeline()

# Run feature engineering
result = pipeline.run(
    data=data,
    target_column="target",
    problem_description="Predict customer churn based on usage patterns",
    categorical_columns=["plan_type", "region"],
    is_classification=True,
)

# Access results
transformed_data = result.transformed_data
selected_features = result.selected_features
feature_metadata = result.feature_metadata
```

## Project Structure

```
faster/
├── __init__.py                 # Package initialization
├── domain_knowledge.py         # Domain knowledge extraction
├── feature_generation.py       # Feature generation and transformation
├── statistical_evaluation.py   # Statistical testing
├── feature_selection.py        # Feature selection
├── pipeline.py                # Main pipeline orchestration
└── utils/                     # Utility functions
    ├── __init__.py
    ├── logging.py
    └── validation.py
```

## Configuration

The pipeline can be configured through the `PipelineConfig` class:

```python
from faster import Pipeline
from faster.pipeline import PipelineConfig
from faster.feature_selection import SelectionCriteria

config = PipelineConfig(
    model_name="gpt-4",
    temperature=0.0,
    max_interaction_degree=2,
    alpha=0.05,
    selection_criteria=SelectionCriteria(
        p_value_threshold=0.05,
        min_effect_size=0.1,
        max_correlation=0.9,
    ),
    output_dir="results",
    save_intermediate=True,
)

pipeline = Pipeline(config)
```

## Feature Generation

FASTER supports various feature transformation types:

- Basic transformations (scaling, normalization)
- Interaction features
- Domain-specific transformations
- Text feature extraction
- Time-based features

## Statistical Evaluation

Features are evaluated using:

- Correlation analysis
- Effect size calculation
- Statistical significance testing
- Mutual information scores
- Multiple testing correction

## Feature Selection

Selection criteria include:

- Statistical significance
- Effect size
- Mutual information
- Feature importance from ML models
- Correlation analysis

## Output

The pipeline generates:

- Transformed dataset
- Feature importance scores
- Statistical metrics
- Feature metadata
- Execution logs

## Development

```bash
# Install development dependencies
rye sync --dev

# Run tests
pytest

# Run linting
ruff check .
```

## Requirements

- Python 3.9+
- Dependencies listed in pyproject.toml

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
