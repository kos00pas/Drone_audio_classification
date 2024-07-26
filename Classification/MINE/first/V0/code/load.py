import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import pandas as pd
import h5py

print("Loading CSV file...")

# Load the CSV file containing paths and labels
file_path = 'final_updated_all_paths_and_labels.csv'
data = pd.read_csv(file_path)

# Extract file paths and labels
file_paths = data.iloc[:, 0].values
labels = data.iloc[:, 2].values

print(f"Total samples: {len(file_paths)}")

# Create a TensorFlow dataset from the file paths and labels
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

@tf.function
def load_mfcc(file_path):
    file_path = tf.strings.regex_replace(file_path, '\\\\', '/')  # Replace backslashes with forward slashes
    mfcc_content = tf.io.read_file(file_path)
    mfcc_lines = tf.strings.split(mfcc_content, '\n')

    # Filter out empty lines
    mfcc_lines = tf.boolean_mask(mfcc_lines, tf.strings.length(mfcc_lines) > 0)

    # Split lines by comma and convert to float
    mfcc_lines = tf.map_fn(
        lambda x: tf.strings.to_number(tf.strings.split(x, ','), out_type=tf.float32),
        mfcc_lines,
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.float32)
    )
    mfcc_tensor = mfcc_lines.to_tensor()

    return mfcc_tensor

@tf.function
def filter_mfcc(mfcc_tensor, label):
    shape = tf.shape(mfcc_tensor)
    return tf.reduce_all(tf.equal(shape, [40, 32]))

# Map the dataset to load the MFCC files
dataset = dataset.map(lambda x, y: (load_mfcc(x), y))

# Filter out samples that don't have the shape (40, 32)
dataset = dataset.filter(lambda x, y: filter_mfcc(x, y))

# Calculate the sizes of each split
dataset_size = len(file_paths)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

# Split the dataset
train_dataset = dataset.take(train_size)
val_test_dataset = dataset.skip(train_size)

# Check contents of val_test_dataset
def debug_dataset_contents(dataset, name):
    count = 0
    for i, (mfcc, label) in enumerate(dataset):
        if i < 5:  # Print first few samples for debugging
            print(f"{name} sample {i}: MFCC shape {mfcc.shape}, label {label.numpy()}")
        count += 1
    print(f"{name} dataset size: {count}")
    return count

val_test_count = debug_dataset_contents(val_test_dataset, "Val/Test")

# Further split val_test_dataset into validation and test datasets
val_dataset = val_test_dataset.take(val_size)
test_dataset = val_test_dataset.skip(val_size)

# Ensure there are elements in the datasets
def count_elements(dataset, name):
    count = 0
    for _ in dataset:
        count += 1
    print(f"{name} dataset size: {count}")
    return count

train_count = count_elements(train_dataset, "Train")
val_count = count_elements(val_dataset, "Validation")
test_count = count_elements(test_dataset, "Test")

# Save datasets using h5py
def save_to_h5(dataset, file_name):
    with h5py.File(file_name, 'w') as f:
        mfcc_grp = f.create_group('mfcc')
        label_grp = f.create_group('label')
        for i, (mfcc, label) in enumerate(dataset):
            mfcc_np = mfcc.numpy()
            label_np = label.numpy()
            if mfcc_np.shape != (40, 32):
                print(f"Skipping sample with incorrect shape: {mfcc_np.shape}")
                continue
            mfcc_grp.create_dataset(str(i), data=mfcc_np)
            label_grp.create_dataset(str(i), data=label_np)
            if i < 5:  # Print first few samples for debugging
                print(f"Saved sample {i}: MFCC shape {mfcc_np.shape}, label {label_np}")

if train_count > 0:
    save_to_h5(train_dataset, 'train_dataset.h5')
if val_count > 0:
    save_to_h5(val_dataset, 'val_dataset.h5')
if test_count > 0:
    save_to_h5(test_dataset, 'test_dataset.h5')

print("Datasets saved to HDF5")
