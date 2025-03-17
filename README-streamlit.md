# FASTER Pipeline Demo

This Streamlit application demonstrates the **Feature Automation, Selection, Transformation, Extraction Routine (FASTER)** pipeline for automated feature engineering. The pipeline utilizes domain knowledge from large language models to enhance model performance.

## Features

- Interactive demonstration of the FASTER pipeline
- Selection from predefined datasets (Iris, Auto MPG, Titanic, Horsepower-MPG)
- Custom dataset upload
- Configuration of LLM and pipeline parameters
- Detailed performance metrics and visualizations
- Feature importance analysis

## Installation

### Using Rye (Recommended)

1. Make sure you have [Rye](https://github.com/astral-sh/rye) installed
2. Clone the repository
3. Navigate to the project directory
4. Install the project dependencies:

```bash
rye sync
```

### Using pip

1. Clone the repository
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install -r requirements-streamlit.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open the provided URL in your web browser (typically http://localhost:8501)

3. Select a dataset or upload your own

4. Configure the pipeline parameters:
   - LLM model
   - Statistical significance level
   - Interaction degree
   - Multiple testing correction method

5. Click "Run FASTER Pipeline" to start the analysis

6. View the results in the various tabs:
   - Performance Metrics
   - Feature Analysis
   - Data Transformation
   - Pipeline Details

## OpenRouter API Key

To use real LLM responses instead of mock data, you need to provide an OpenRouter API key. You can get a key by signing up at [OpenRouter](https://openrouter.ai/).

## Datasets

The app comes with four predefined datasets:

1. **Iris** - Regression problem predicting sepal length
2. **Auto MPG** - Regression problem predicting fuel efficiency
3. **Titanic** - Classification problem predicting survival
4. **Horsepower-MPG** - Simple regression problem demonstrating non-linear relationships

## Notes

- The first run may take longer due to model loading and initial data processing
- Running with real LLM responses requires an OpenRouter API key and may incur costs
- For large custom datasets, the processing time may be significant

## License

MIT License 