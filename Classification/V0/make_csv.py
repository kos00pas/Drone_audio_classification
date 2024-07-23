import os
import pandas as pd

# Get the current directory
current_directory = os.getcwd()

# Set to store directories containing CSV files
directories_with_csv = set()

# Walk through the directory and its subfolders
for root, dirs, files in os.walk(current_directory):
    for file in files:
        if file.endswith('.csv'):
            # Calculate the relative directory path and add to the set
            relative_dir = os.path.relpath(root, current_directory)
            directories_with_csv.add('.\\' + relative_dir)

# Prepare the lists for the CSV columns
directories = []
signals = []
mfccs = []
labels = []

# Prepare lists for the second CSV
mfcc_paths = []
label_data = []

for directory in sorted(directories_with_csv):
    directories.append(directory)

    signal_path = None
    mfcc_path = None
    label_path = None

    # Search for signal.csv, mfcc.csv, and label.csv in the directory
    for root, dirs, files in os.walk(os.path.join(current_directory, directory)):
        for file in files:
            if file == 'signal.csv':
                signal_path = '.\\' + os.path.relpath(os.path.join(root, file), current_directory)
            elif file == 'mfcc.csv':
                mfcc_path = '.\\' + os.path.relpath(os.path.join(root, file), current_directory)
            elif file == 'label.csv':
                label_path = '.\\' + os.path.relpath(os.path.join(root, file), current_directory)

    signals.append(signal_path if signal_path else '')
    mfccs.append(mfcc_path if mfcc_path else '')
    labels.append(label_path if label_path else '')

    # Add paths to the second CSV lists
    mfcc_paths.append(mfcc_path if mfcc_path else '')

    # Read the [0][0] data from the label.csv
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            first_data = first_line.split(',')[0] if first_line else ''
            if first_data == 'drone':
                label_data.append((1, 'drone'))
            elif first_data == 'not_drone':
                label_data.append((0, 'not_drone'))
            else:
                label_data.append(('', ''))
    else:
        label_data.append(('', ''))

# Create the first DataFrame and save to CSV
df1 = pd.DataFrame({
    'directories': directories,
    'signals': signals,
    'mfccs': mfccs,
    'labels': labels
})
df1.to_csv('directories_info.csv', index=False)

# Create the second DataFrame and save to CSV
df2 = pd.DataFrame({
    'mfccs': mfcc_paths,
    'label_value': [item[0] for item in label_data],
    'label_text': [item[1] for item in label_data]
})
df2.to_csv('mfccs_labels.csv', index=False)

print("CSV files created: directories_info.csv and mfccs_labels.csv")
