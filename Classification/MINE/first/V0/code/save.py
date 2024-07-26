import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import pandas as pd
from joblib import dump


print("Done loading ")

# Load the CSV file containing paths and labels
file_path = 'final_updated_all_paths_and_labels.csv'
data = pd.read_csv(file_path)

# Extract file paths and labels
file_paths = data.iloc[:, 0].values
labels = data.iloc[:, 2].values

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
val_dataset = val_test_dataset.take(val_size)
test_dataset = val_test_dataset.skip(val_size)

# Save datasets using joblib
def save_to_joblib(dataset, file_name):
    data_list = []
    for mfcc, label in dataset:
        mfcc = mfcc.numpy()
        label = label.numpy()
        data_list.append((mfcc, label))
    dump(data_list, file_name)

save_to_joblib(train_dataset, 'train_dataset.joblib')
save_to_joblib(val_dataset, 'val_dataset.joblib')
save_to_joblib(test_dataset, 'test_dataset.joblib')

print("Datasets saved to joblib")
