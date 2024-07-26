import os
import pandas as pd
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import h5py


# Load the labels and file paths from mfccs_labels.csv
labels_file_path = 'mfccs_labels.csv'  # Update with the actual path
labels_data = pd.read_csv(labels_file_path, header=None, names=['mfccs', 'label_value'])

# Define the expected shape
expected_shape = (40, 32)

# Function to check the shape of each MFCC file
def check_mfcc_shape(file_path):
    try:
        if isinstance(file_path, str) and os.path.exists(file_path):
            mfcc_data = pd.read_csv(file_path, header=None)
            shape = mfcc_data.shape
            if shape == expected_shape:
                return True, mfcc_data
            else:
                print(f"ERROR: {file_path} has incorrect shape: {shape}. Expected: {expected_shape}")
                return False, None
        else:
            print(f"ERROR: Invalid file path: {file_path}")
            return False, None
    except Exception as e:
        print(f"ERROR: Could not read {file_path}. Exception: {e}")
        return False, None

# Initialize lists to hold valid MFCC data and labels
valid_mfcc_data = []
valid_labels = []

# Iterate through each path in mfccs_labels.csv and check the MFCC shape
for index, row in labels_data.iterrows():
    mfcc_file_path = row['mfccs']
    label = row['label_value']
    is_valid, mfcc_data = check_mfcc_shape(mfcc_file_path)
    if is_valid:
        valid_mfcc_data.append(mfcc_data.values)
        valid_labels.append(label)

# Convert lists to numpy arrays
valid_mfcc_data = np.array(valid_mfcc_data)
valid_labels = np.array(valid_labels)

# Print the total number of valid samples
print(f"Total valid samples: {len(valid_mfcc_data)}")

# Create a TensorFlow dataset from the valid MFCC data and labels
mfcc_dataset = tf.data.Dataset.from_tensor_slices((valid_mfcc_data, valid_labels))

# Shuffle and split the dataset into training, validation, and test sets
dataset_size = len(valid_mfcc_data)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = mfcc_dataset.take(train_size)
val_test_dataset = mfcc_dataset.skip(train_size)
val_dataset = val_test_dataset.take(val_size)
test_dataset = val_test_dataset.skip(val_size)

# Print the sizes of each dataset
print(f"Train dataset size: {train_size}")
print(f"Validation dataset size: {val_size}")
print(f"Test dataset size: {test_size}")

# Function to save datasets using h5py
def save_to_h5(dataset, file_name):
    with h5py.File(file_name, 'w') as f:
        mfcc_grp = f.create_group('mfcc')
        label_grp = f.create_group('label')
        for i, (mfcc, label) in enumerate(dataset):
            mfcc_np = mfcc.numpy()
            label_np = label.numpy()
            mfcc_grp.create_dataset(str(i), data=mfcc_np)
            label_grp.create_dataset(str(i), data=label_np)

# Save the datasets to HDF5 files
save_to_h5(train_dataset, 'train_dataset.h5')
save_to_h5(val_dataset, 'val_dataset.h5')
save_to_h5(test_dataset, 'test_dataset.h5')

print("Datasets saved to HDF5")

# Function to count labels in an HDF5 file
def count_labels_in_h5(file_name):
    with h5py.File(file_name, 'r') as f:
        labels = [f['label'][key][()] for key in f['label'].keys()]
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        return label_counts

# Count labels in each HDF5 file
train_label_counts = count_labels_in_h5('train_dataset.h5')
val_label_counts = count_labels_in_h5('val_dataset.h5')
test_label_counts = count_labels_in_h5('test_dataset.h5')

# Print label counts
print(f"Train dataset label counts: {train_label_counts}")
print(f"Validation dataset label counts: {val_label_counts}")
print(f"Test dataset label counts: {test_label_counts}")
