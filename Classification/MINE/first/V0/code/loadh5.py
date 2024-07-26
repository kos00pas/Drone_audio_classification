import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import h5py
import tensorflow as tf

def load_dataset_from_h5(file_name):
    with h5py.File(file_name, 'r') as f:
        mfcc_grp = f['mfcc']
        label_grp = f['label']
        data_list = [(mfcc_grp[key][()], label_grp[key][()]) for key in mfcc_grp.keys()]

    def generator():
        for mfcc, _ in data_list:  # Ignore the label by using '_'
            yield mfcc

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(40, 32), dtype=tf.float32)
        )
    )
    return dataset, len(data_list)  # Return the dataset and its size

# Load the datasets
train_dataset, train_size = load_dataset_from_h5('train_dataset.h5')
val_dataset, val_size = load_dataset_from_h5('val_dataset.h5')
test_dataset, test_size = load_dataset_from_h5('test_dataset.h5')

# Print dataset sizes
print(f"Training dataset size: {train_size}")
print(f"Validation dataset size: {val_size}")
print(f"Testing dataset size: {test_size}")

# Example: Iterate through the dataset and print shapes
for mfcc in train_dataset.take(1):
    print("MFCC shape:", mfcc.shape)
