import os
import pandas as pd

def print_mfcc_shapes(root_dir):
    """
    Traverse through each subdirectory in the root directory to find mfcc.csv
    and print the shape of the MFCC data if it is different from (40, 32).
    """
    print("start running, wait")
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'mfcc.csv':
                file_path = os.path.join(subdir, file)
                mfcc_data = pd.read_csv(file_path)
                if mfcc_data.shape != (40, 32):
                    print(f'Shape of {file_path}: {mfcc_data.shape}')

# Set the root directory
all_data = './all_data'

# Example usage
print_mfcc_shapes(all_data)
