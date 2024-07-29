import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras import layers, models
import h5py
import numpy as np
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_h5_dataset(file_name):
    print("Loading h5 file...")

    with h5py.File(file_name, 'r') as f:
        mfcc_data = []
        labels = []
        for key in f['mfcc'].keys():
            mfcc_data.append(f['mfcc'][key][()])
            labels.append(f['label'][key][()])

        mfcc_data = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int64)

        dataset = tf.data.Dataset.from_tensor_slices((mfcc_data, labels))
    print("Loaded h5 file successfully")
    return dataset


def prepare_datasets(train_file, val_file, test_file, batch_size=32, shuffle_buffer_size=1000):
    print("Preparing datasets...")

    train_dataset = load_h5_dataset(train_file)
    val_dataset = load_h5_dataset(val_file)
    test_dataset = load_h5_dataset(test_file)

    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Datasets prepared successfully")
    return train_dataset, val_dataset, test_dataset


def create_model(learning_rate):
    print("Creating model...")

    model = models.Sequential([
        layers.Input(shape=(40, 32, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("Model created successfully")
    return model


if __name__ == "__main__":
    print("Starting main process...")
    # Define hyperparameter ranges
    learning_rates = [0.001, 0.0001, 0.00001]
    batch_sizes = [32, 64, 128]
    epochs = [10, 20, 30]

    train_file = 'train_dataset.h5'
    val_file = 'val_dataset.h5'
    test_file = 'test_dataset.h5'

    print("Loading and preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_file, val_file, test_file)
    print("Datasets are ready")

    # Directory to save models and outputs
    output_dir = 'model_outputs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # List to store results
    results = []

    # Loop through all combinations
    combination_number = 0
    total_combinations = len(learning_rates) * len(batch_sizes) * len(epochs)
    print(f"Total combinations to run: {total_combinations}")

    for lr in learning_rates:
        for bs in batch_sizes:
            for ep in epochs:
                combination_number += 1
                print(f"Running combination {combination_number}/{total_combinations} (LR: {lr}, BS: {bs}, Ep: {ep})")

                # Create model
                model = create_model(lr)

                # Setup output file
                model_name = f'trained_model_LR_{lr}_BS_{bs}_Ep_{ep}'
                output_file = os.path.join(output_dir, f'{model_name}.txt')

                # Redirect stdout to a file with utf-8 encoding
                with open(output_file, 'w', encoding='utf-8') as f:
                    sys.stdout = f

                    # Print hyperparameters
                    print(f"Training with Learning Rate: {lr}, Batch Size: {bs}, Epochs: {ep}")

                    # Train model
                    print("Starting model training...")
                    history = model.fit(train_dataset, epochs=ep, validation_data=val_dataset, verbose=1)
                    print("Model training completed")

                    # Evaluate model
                    print("Evaluating model on validation dataset...")
                    val_loss, val_accuracy = model.evaluate(val_dataset)
                    print(f"Validation Accuracy: {val_accuracy}")
                    print(f"Validation Loss: {val_loss}")

                    print("Evaluating model on test dataset...")
                    test_loss, test_accuracy = model.evaluate(test_dataset)
                    print(f"Test Accuracy: {test_accuracy}")
                    print(f"Test Loss: {test_loss}")

                    # Save model
                    print(f"Saving model: {model_name}.keras")
                    model.save(os.path.join(output_dir, f'{model_name}.keras'))

                    # Store results
                    results.append({
                        'learning_rate': lr,
                        'batch_size': bs,
                        'epochs': ep,
                        'val_accuracy': val_accuracy,
                        'val_loss': val_loss,
                        'test_accuracy': test_accuracy,
                        'test_loss': test_loss
                    })

                # Revert stdout
                sys.stdout = sys.__stdout__

                print(f"Combination {combination_number}/{total_combinations} completed.")

    print("All combinations completed. Saving summary results...")
    # Save results to a summary file
    summary_file = os.path.join(output_dir, 'summary_results.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"LR: {result['learning_rate']}, BS: {result['batch_size']}, Ep: {result['epochs']}, "
                    f"Val Acc: {result['val_accuracy']:.4f}, Val Loss: {result['val_loss']:.4f}, "
                    f"Test Acc: {result['test_accuracy']:.4f}, Test Loss: {result['test_loss']:.4f}\n")

    print("Training complete. Models and logs are saved in 'model_outputs' directory.")
    print("Summary of all runs saved in 'summary_results.txt'")
