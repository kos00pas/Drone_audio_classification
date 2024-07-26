import os
import pandas as pd
import csv

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("all_paths_and_labels.csv")

# Extract directory column into a list
directories = df['Directory'].tolist()

# Extract labels column into a list
labels = df['Label'].tolist()

def create_label_csv(directory, label):
    label_file_path = os.path.join(directory, "label.csv")
    with open(label_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([label])

for directory, label in zip(directories, labels):
    if os.path.exists(directory):
        if label == 'drone':
            x_string = 'drone'
            create_label_csv(directory, x_string)
        else:
            x_string = 'not_drone'
            create_label_csv(directory, x_string)
    else:
        print(f"Directory '{directory}' does not exist. Skipping...")
