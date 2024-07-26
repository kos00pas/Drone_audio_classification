import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras import layers, models
import h5py


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

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    """Input Shape: Ensure your input data shape matches (40, 32, 1). It is suitable for images with one channel (grayscale).
    Layers and Filters: The chosen layers and filters (Conv2D) are appropriate for extracting features.
    Dense Layers: The Dense layers after flattening are suitable for classification tasks.
    Activation Function: Using 'sigmoid' for the final Dense layer is correct for binary classification.
    Loss Function: 'binary_crossentropy' is the right choice for binary classification.
    Optimizer and Metrics: 'adam' optimizer and 'accuracy' metric are good choices."""
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

"""C:\Users\kos00\anaconda3\envs\doa_env\python.exe C:\Users\kos00\Documents\Run_programs\KIOS\Drone_audio_classification\Classification\MINE\second\v0\loadh5.py 
start
done prepare_datasets
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 38, 30, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 19, 15, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 17, 13, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 8, 6, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 6, 4, 64)       │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 1536)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │        98,368 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 154,177 (602.25 KB)
 Trainable params: 154,177 (602.25 KB)
 Non-trainable params: 0 (0.00 B)
lets fit
Epoch 1/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 9s 26ms/step - accuracy: 0.8237 - loss: 0.4544 - val_accuracy: 0.9975 - val_loss: 0.0164
Epoch 2/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 8s 28ms/step - accuracy: 0.8741 - loss: 0.3286 - val_accuracy: 0.9907 - val_loss: 0.0365
Epoch 3/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 8s 27ms/step - accuracy: 0.8986 - loss: 0.2488 - val_accuracy: 0.9980 - val_loss: 0.0150
Epoch 4/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 8s 27ms/step - accuracy: 0.9095 - loss: 0.2348 - val_accuracy: 0.9961 - val_loss: 0.0244
Epoch 5/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.9169 - loss: 0.2187 - val_accuracy: 0.9980 - val_loss: 0.0125
Epoch 6/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.9169 - loss: 0.2320 - val_accuracy: 0.9956 - val_loss: 0.0147
Epoch 7/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 8s 28ms/step - accuracy: 0.9248 - loss: 0.1985 - val_accuracy: 0.9921 - val_loss: 0.0354
Epoch 8/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 8s 28ms/step - accuracy: 0.9250 - loss: 0.1904 - val_accuracy: 0.9931 - val_loss: 0.0287
Epoch 9/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.9362 - loss: 0.1704 - val_accuracy: 0.9887 - val_loss: 0.0389
Epoch 10/10
297/297 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.9287 - loss: 0.1780 - val_accuracy: 0.9966 - val_loss: 0.0134
lets loss
64/64 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.4734 - loss: 6.9151 
Test Accuracy: 0.6247544288635254
Model saved to trained_model_20240726_174822.keras
Model loaded from trained_model_20240726_174822.keras
64/64 ━━━━━━━━━━━━━━━━━━━━ 1s 12ms/step - accuracy: 0.4734 - loss: 6.9151
Test Accuracy of the loaded model: 0.6247544288635254

Process finished with exit code 0
"""