import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import pandas as pd
import h5py

print("Loading CSV file...")

# Load the CSV file containing paths and labels
file_path = 'mfccs_labels.csv'
data = pd.read_csv(file_path)

# Extract file paths and labels
file_paths = data.iloc[:, 0].values
labels = data.iloc[:, 1].values

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
    expected_shape = tf.constant([40, 32])

    # Check if the shapes are different
    shape_check = tf.reduce_all(tf.equal(shape, expected_shape))

    # Use tf.print for logging inside a tf.function
    # tf.print("Shape of mfcc_tensor:", shape)

    # Print only if shapes are different
    tf.cond(
        tf.logical_not(shape_check),
        lambda: tf.print("Different shape:", shape),
        lambda: tf.no_op()
    )

    return shape_check

# Map the dataset to load the MFCC files
dataset = dataset.map(lambda x, y: (load_mfcc(x), y))
print("Done load_mfcc")

# Filter out samples that don't have the shape (40, 32)
dataset = dataset.filter(lambda x, y: filter_mfcc(x, y))
print("Done filter")

# Count the elements in the filtered dataset
def count_elements(dataset):
    return dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()

# Calculate the size of the filtered dataset
filtered_dataset_size = count_elements(dataset)
print(f"Filtered dataset size: {filtered_dataset_size}")

# Calculate the sizes of each split
train_size = int(0.7 * filtered_dataset_size)
val_size = int(0.15 * filtered_dataset_size)
test_size = filtered_dataset_size - train_size - val_size

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=filtered_dataset_size, reshuffle_each_iteration=False)

# Split the dataset
train_dataset = dataset.take(train_size)
val_test_dataset = dataset.skip(train_size)
val_dataset = val_test_dataset.take(val_size)
test_dataset = val_test_dataset.skip(val_size)

# Ensure there are elements in the datasets
print(f"Train dataset size: {count_elements(train_dataset)}")
print(f"Validation dataset size: {count_elements(val_dataset)}")
print(f"Test dataset size: {count_elements(test_dataset)}")

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

save_to_h5(train_dataset, 'train_dataset.h5')
save_to_h5(val_dataset, 'val_dataset.h5')
save_to_h5(test_dataset, 'test_dataset.h5')

print("Datasets saved to HDF5")
