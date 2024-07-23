import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
import h5py

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

def load_h5_dataset(file_name):
    with h5py.File(file_name, 'r') as f:
        mfcc_data = []
        labels = []
        for key in f['mfcc'].keys():
            mfcc_data.append(f['mfcc'][key][()])
            labels.append(f['label'][key][()])

        mfcc_data = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)

        dataset = tf.data.Dataset.from_tensor_slices((mfcc_data, labels))
    return dataset


def prepare_datasets(train_file, val_file, test_file, batch_size=32, shuffle_buffer_size=1000):
    train_dataset = load_h5_dataset(train_file)
    val_dataset = load_h5_dataset(val_file)
    test_dataset = load_h5_dataset(test_file)

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_file = 'train_dataset.h5'
    val_file = 'val_dataset.h5'
    test_file = 'test_dataset.h5'

    print("start")
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_file, val_file, test_file)
    print("done prepare_datasets")

    # Define a simple CNN model for demonstration
    model = models.Sequential([
        layers.Input(shape=(40, 32, 1)),  # Adding a channel dimension for CNN
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid', dtype=tf.float32)  # Use float32 for the final layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

    print("lets fit")
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset
    )

    print("lets loss")
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc}")

    from datetime import datetime

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the model name dynamically using the timestamp
    model_name = f'trained_model_{timestamp}.keras'

    # Save the model in Keras format
    model.save(model_name)
    print(f"Model saved to {model_name}")

    # Load the model
    loaded_model = models.load_model(model_name)
    print(f"Model loaded from {model_name}")

    # Evaluate the loaded model
    test_loss, test_acc = loaded_model.evaluate(test_dataset)
    print(f"Test Accuracy of the loaded model: {test_acc}")
