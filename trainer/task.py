"""Main training module for Vertex AI custom training."""

import os
import argparse
import logging
import tensorflow as tf
import hypertune
import pandas as pd
import json
from datetime import datetime

from trainer import model
from trainer import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to the training data file (can be a GCS path)'
    )
    parser.add_argument(
        '--target-column',
        type=str,
        default='target',
        help='Name of the target column in the dataset'
    )
    parser.add_argument(
        '--categorical-columns',
        type=str,
        help='Comma-separated list of categorical columns to one-hot encode'
    )
    
    # Model arguments
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train'
    )
    
    # Output arguments
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Directory to save the model (if not specified, will use AIP_MODEL_DIR)'
    )
    
    # Hyperparameter tuning
    parser.add_argument(
        '--hp-tune',
        action='store_true',
        help='Enable hyperparameter tuning reporting'
    )
    
    return parser.parse_args()

def train_and_evaluate(args):
    """Train and evaluate the model."""
    try:
        # Check if running on Vertex AI with managed datasets
        if 'AIP_TRAINING_DATA_URI' in os.environ:
            logger.info("Using Vertex AI managed datasets")
            data_paths = utils.get_data_from_env_vars()
            
            # Load training data
            if data_paths['training']:
                logger.info(f"Loading training data from {data_paths['training']}")
                train_df = utils.load_data_from_gcs(data_paths['training'])
            else:
                raise ValueError("No training data specified")
            
            # Load validation data if available
            if data_paths['validation']:
                logger.info(f"Loading validation data from {data_paths['validation']}")
                val_df = utils.load_data_from_gcs(data_paths['validation'])
            else:
                val_df = None
                
            # Preprocess training data
            categorical_columns = args.categorical_columns.split(',') if args.categorical_columns else None
            
            # Handle target column
            if args.target_column not in train_df.columns:
                logger.error(f"Target column {args.target_column} not found in training data")
                logger.info(f"Available columns: {list(train_df.columns)}")
                raise ValueError(f"Target column {args.target_column} not found in training data")
                
            # Preprocess training data
            if val_df is not None:
                # Use separate validation set
                X_train = train_df.drop(columns=[args.target_column])
                y_train = train_df[args.target_column]
                X_val = val_df.drop(columns=[args.target_column])
                y_val = val_df[args.target_column]
            else:
                # Split training data to create validation set
                X_train, y_train, X_val, y_val = utils.preprocess_data(
                    train_df, args.target_column, categorical_columns
                )
        else:
            # Running locally or with explicit data path
            if not args.data_path:
                raise ValueError("--data-path must be specified when not using Vertex AI managed datasets")
                
            logger.info(f"Loading data from {args.data_path}")
            
            # Check if GCS path or local path
            if args.data_path.startswith('gs://'):
                df = utils.load_data_from_gcs(args.data_path)
            else:
                df = pd.read_csv(args.data_path)
                
            # Handle categorical columns
            categorical_columns = args.categorical_columns.split(',') if args.categorical_columns else None
            
            # Preprocess and split the data
            X_train, y_train, X_val, y_val = utils.preprocess_data(
                df, args.target_column, categorical_columns
            )
        
        # Get output directory for model
        if args.model_dir:
            model_dir = args.model_dir
        else:
            model_dir = utils.get_model_dir()
        
        logger.info(f"Model will be saved to {model_dir}")
        
        # Get feature dimensionality and number of classes
        input_shape = X_train.shape[1]
        num_classes = len(pd.Series(y_train).unique())
        
        logger.info(f"Input shape: {input_shape}, Number of classes: {num_classes}")
        
        # Create and compile the model
        ml_model = model.create_model(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=args.learning_rate
        )
        
        # Create TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
        
        # Create checkpoint callback
        checkpoint_path = os.path.join(model_dir, 'checkpoints', 'model.ckpt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy'
        )
        
        # Add hyperparameter tuning callback if needed
        callbacks = [tensorboard_callback, checkpoint_callback]
        
        if args.hp_tune:
            hp_callback = tf.keras.callbacks.Callback()
            def on_epoch_end(epoch, logs=None):
                hpt = hypertune.HyperTune()
                hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag='accuracy',
                    metric_value=logs['val_accuracy'],
                    global_step=epoch
                )
            hp_callback.on_epoch_end = on_epoch_end
            callbacks.append(hp_callback)
        
        # Train the model
        logger.info("Starting model training...")
        history = ml_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks
        )
        
        # Save training metrics
        metrics_path = os.path.join(model_dir, 'metrics.json')
        with tf.io.gfile.GFile(metrics_path, 'w') as f:
            metrics = {
                'accuracy': float(history.history['accuracy'][-1]),
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'loss': float(history.history['loss'][-1]),
                'val_loss': float(history.history['val_loss'][-1])
            }
            json.dump(metrics, f)
        
        # Save the model in SavedModel format for TensorFlow Serving
        saved_model_path = os.path.join(model_dir, 'saved_model')
        logger.info(f"Saving model to {saved_model_path}")
        ml_model.save(saved_model_path)
        
        # Also save as HDF5 format for easier loading
        h5_path = os.path.join(model_dir, 'model.h5')
        ml_model.save(h5_path)
        
        logger.info("Training completed successfully!")
        return metrics
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
        
def main():
    """Main function for training."""
    args = get_args()
    
    # Log important arguments
    logger.info(f"Training with batch size: {args.batch_size}, learning rate: {args.learning_rate}")
    
    # Train and evaluate the model
    metrics = train_and_evaluate(args)
    
    logger.info(f"Final model metrics: {metrics}")

if __name__ == '__main__':
    main()