"""Utility functions for data processing."""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_from_gcs(data_path):
    """Load data from a Google Cloud Storage path.
    
    Args:
        data_path: Path to data file or directory in GCS
        
    Returns:
        Loaded data as a pandas DataFrame
    """
    # For Cloud Storage paths, use tf.io.gfile
    with tf.io.gfile.GFile(data_path, 'rb') as f:
        return pd.read_csv(f)

def get_data_from_env_vars():
    """Get data paths from Vertex AI environment variables.
    
    Returns:
        Dictionary with paths to training, validation, and test data
    """
    # Vertex AI sets these environment variables for managed datasets
    training_data_uri = os.environ.get('AIP_TRAINING_DATA_URI')
    validation_data_uri = os.environ.get('AIP_VALIDATION_DATA_URI')
    test_data_uri = os.environ.get('AIP_TEST_DATA_URI')
    
    # Get the data format
    data_format = os.environ.get('AIP_DATA_FORMAT', 'csv')
    
    return {
        'training': training_data_uri,
        'validation': validation_data_uri,
        'test': test_data_uri,
        'format': data_format
    }

def get_model_dir():
    """Get the model directory from environment variables.
    
    Returns:
        Path where the model should be saved
    """
    # Vertex AI provides this environment variable
    model_dir = os.environ.get('AIP_MODEL_DIR')
    
    # If running locally without the env var, use a default
    if model_dir is None:
        model_dir = 'trained_model'
        os.makedirs(model_dir, exist_ok=True)
        
    return model_dir

def preprocess_data(df, target_column, categorical_columns=None, test_size=0.2):
    """Preprocess data for training.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        categorical_columns: List of categorical columns to one-hot encode
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical features if specified
    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return X_train, y_train, X_test, y_test