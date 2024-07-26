import os
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model_path = '../MODEL/trained_model_20240723_181144.keras'
model = load_model(model_path)
print(f"Model loaded from {model_path}")
# Function to load and preprocess MFCC data from a CSV file
def load_mfcc_from_csv(file_path, expected_shape=(40, 32)):
    try:
        mfcc_data = pd.read_csv(file_path, header=None)
        if mfcc_data.shape == expected_shape:
            mfcc_data = np.expand_dims(mfcc_data.values, axis=-1)  # Add channel dimension
            return tf.convert_to_tensor(mfcc_data, dtype=tf.float32)
        else:
            print(f"ERROR: {file_path} has incorrect shape: {mfcc_data.shape}. Expected: {expected_shape}")
            return None
    except Exception as e:
        print(f"ERROR: Could not read {file_path}. Exception: {e}")
        return None

# Load the MFCC data from the CSV file
mfcc_csv_path = 'path/to/your/mfcc.csv'
mfcc_tensor = load_mfcc_from_csv(mfcc_csv_path)

if mfcc_tensor is not None:
    mfcc_tensor = tf.expand_dims(mfcc_tensor, axis=0)  # Add batch dimension
    predictions = model.predict(mfcc_tensor)
    print(f"Predictions: {predictions}")
else:
    print("Failed to load MFCC data from CSV file.")
