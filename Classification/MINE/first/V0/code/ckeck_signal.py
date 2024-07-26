import csv
import pandas as pd

def extract_paths_from_csv(file_path):
    paths = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Check if row is not empty
                paths.append(row[0])
    return paths


def get_csv_shape(file_path):
    csv_files_to_check = [
        'mfcc_all_paths_and_labels.csv',
        'not_drone.csv',
        'all_paths_and_labels.csv',
        'drone.csv'
    ]

    try:
        df = pd.read_csv(file_path)
        return df.shape
    except FileNotFoundError as e:
        print(file_path)
        pass
        # Handle file not found error by deleting the line from other CSV files
        for csv_file in csv_files_to_check:
            try:
                with open(csv_file, 'r') as file:
                    lines = file.readlines()
                with open(csv_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    for line in lines:
                        if file_path not in line:
                            file.write(line)
            except Exception as inner_e:
                print(f"Error handling file {csv_file}: {inner_e}")
        return f"[Errno 2] No such file or directory: '{file_path}'"
    except Exception as e:
        return str(e)

# Example usage
file_path = "signal_all_paths_and_labels.csv"
paths = extract_paths_from_csv(file_path)

for path in paths:
    shape = get_csv_shape(path)
    if shape != (15999, 1):
        pass
        # print(f"File: {path} Shape: {shape}")
