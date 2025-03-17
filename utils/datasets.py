"""Dataset loading utilities for FASTER Streamlit app."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Iris dataset for regression (predicting sepal length)."""
    # Load Iris dataset
    iris = datasets.load_iris()
    
    # Create DataFrame
    feature_names = iris.feature_names
    df = pd.DataFrame(iris.data, columns=feature_names)
    df['species'] = iris.target
    
    # Map species to string labels for better interpretability
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_mapping)
    
    # Rename columns to more readable names
    column_mapping = {
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width'
    }
    df = df.rename(columns=column_mapping)
    
    # For regression, we'll predict sepal_length based on other features
    X = df[['sepal_width', 'petal_length', 'petal_width', 'species']].copy()
    y = df['sepal_length']
    
    # Create a combined DataFrame for the pipeline
    data = X.copy()
    data['sepal_length'] = y
    
    return data, y


def load_auto_mpg_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Auto MPG dataset for regression (predicting MPG)."""
    try:
        # Try to fetch the dataset from UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                        'weight', 'acceleration', 'model_year', 'origin', 'car_name']
        df = pd.read_csv(url, delim_whitespace=True, header=None, 
                        names=column_names, na_values='?')
    except Exception as e:
        logger.warning(f"Error fetching from UCI: {str(e)}. Using local data...")
        
        # Use a simplified mock dataset if the real one isn't available
        mpg = np.random.normal(20, 5, 100)
        cylinders = np.random.choice([4, 6, 8], 100)
        displacement = np.random.normal(200, 50, 100)
        horsepower = np.random.normal(100, 30, 100)
        weight = np.random.normal(3000, 500, 100)
        acceleration = np.random.normal(15, 3, 100)
        model_year = np.random.randint(70, 83, 100)
        origin = np.random.choice([1, 2, 3], 100)
        
        df = pd.DataFrame({
            'mpg': mpg,
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model_year': model_year,
            'origin': origin
        })
    
    # Remove car_name column if it exists
    if 'car_name' in df.columns:
        df = df.drop('car_name', axis=1)
    
    # Handle missing values
    numeric_columns = ['horsepower']  # Typically only horsepower has missing values
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Convert 'origin' to a categorical feature with meaningful labels
    origin_mapping = {1: 'american', 2: 'european', 3: 'asian'}
    df['origin'] = df['origin'].map(origin_mapping)
    
    # For regression, we'll predict mpg based on other features
    y = df['mpg']
    X = df.drop('mpg', axis=1)
    
    # Create a combined DataFrame for the pipeline
    data = X.copy()
    data['mpg'] = y
    
    return data, y


def load_titanic_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Titanic dataset for classification (predicting survival)."""
    try:
        # Fetch Titanic dataset from a public URL
        url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
        df = pd.read_csv(url)
    except Exception as e:
        logger.warning(f"Error fetching Titanic dataset: {str(e)}. Using mock data...")
        
        # Create mock Titanic data if fetching fails
        n_samples = 100
        survived = np.random.choice([0, 1], n_samples)
        pclass = np.random.choice([1, 2, 3], n_samples)
        sex = np.random.choice(['male', 'female'], n_samples)
        age = np.random.normal(30, 15, n_samples)
        sibsp = np.random.choice(range(0, 5), n_samples)
        parch = np.random.choice(range(0, 4), n_samples)
        fare = np.random.normal(30, 20, n_samples)
        embarked = np.random.choice(['C', 'Q', 'S'], n_samples)
        
        df = pd.DataFrame({
            'Survived': survived,
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        })
    
    # Basic feature selection
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[['Survived'] + features]
    
    # Handle missing values
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Sex', 'Embarked']
    
    # Impute numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    
    # Impute categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    
    # For classification, we'll predict survival based on other features
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    
    # Create a combined DataFrame for the pipeline
    data = X.copy()
    data['Survived'] = y
    
    return data, y


def load_horsepower_mpg_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Horsepower-MPG dataset (simplified Auto MPG with only horsepower)."""
    try:
        # Try to fetch the Auto MPG dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                        'weight', 'acceleration', 'model_year', 'origin', 'car_name']
        df = pd.read_csv(url, delim_whitespace=True, header=None, 
                        names=column_names, na_values='?')
    except Exception as e:
        logger.warning(f"Error fetching from UCI: {str(e)}. Using local data...")
        
        # Use a simplified mock dataset if the real one isn't available
        mpg = np.random.normal(20, 5, 100)
        horsepower = np.random.normal(100, 30, 100)
        
        df = pd.DataFrame({
            'mpg': mpg,
            'horsepower': horsepower
        })
    
    # Keep only horsepower as feature and mpg as target
    df = df[['mpg', 'horsepower']]
    
    # Handle missing values in horsepower
    imputer = SimpleImputer(strategy='median')
    df['horsepower'] = imputer.fit_transform(df[['horsepower']])
    
    # For regression, we'll predict mpg based on horsepower
    y = df['mpg']
    X = df[['horsepower']]
    
    # Create a combined DataFrame for the pipeline
    data = X.copy()
    data['mpg'] = y
    
    return data, y 