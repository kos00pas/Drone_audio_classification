import os
import pandas as pd

# Load the mfccs_labels.csv file
mfcc_labels_file_path = 'mfccs_labels.csv'
mfcc_labels_data = pd.read_csv(mfcc_labels_file_path)

# Define the expected shape
expected_shape = (40, 32)
def check_mfcc_shape(index, row):
    try:
        mfcc_file_path = row['mfccs']
        if isinstance(mfcc_file_path, str) and os.path.exists(mfcc_file_path):
            mfcc_data = pd.read_csv(mfcc_file_path, header=None)
            shape = mfcc_data.shape
            if shape == expected_shape:
                pass
                """all okay """
                # print(f"{mfcc_file_path} has the correct shape: {shape}")
            else:
                print(f"ERROR: Line {index} - {mfcc_file_path} has incorrect shape: {shape}. Expected: {expected_shape}")
        else:
            print(f"ERROR: Line {index} - Invalid file path: {mfcc_file_path}")
    except Exception as e:
        print(f"ERROR: Line {index} - Could not read {mfcc_file_path}. Exception: {e}")

# Iterate through each path in mfccs_labels.csv and check the MFCC shape
for index, row in mfcc_labels_data.iterrows():
    check_mfcc_shape(index, row)