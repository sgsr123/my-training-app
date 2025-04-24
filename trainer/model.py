"""Model definition for training."""

import tensorflow as tf
from tensorflow import keras

def create_model(input_shape, num_classes, learning_rate=0.001):
    """Creates a classification model.
    
    Args:
        input_shape: The shape of the input data (features)
        num_classes: Number of target classes
        learning_rate: Learning rate for the optimizer
        
    Returns:
        A compiled TensorFlow model
    """
    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # Loss function based on number of classes
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model