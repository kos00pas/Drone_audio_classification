import os
import csv

def find_csv_files_and_record(directory, output_csv_file):
    # Create or open the output CSV file
    print("run")
    with open(output_csv_file, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        # Write headers for the CSV file
        csv_writer.writerow(['Directory', 'Label'])

        # Walk through all directories and files in the specified directory
        for root, dirs, files in os.walk(directory):
            # Check if 'mfcc.csv' is one of the files in the directory
            if 'mfcc.csv' in files:
                # Construct the full path to 'mfcc.csv'
                mfcc_csv_path = os.path.join(root, 'mfcc.csv')
                # Print the full path of 'mfcc.csv'
                #print(mfcc_csv_path)
                # Write the directory and the tag 'drone' to the CSV
                csv_writer.writerow([root, 'drone'])

find_csv_files_and_record('.', 'all_paths_and_labels.csv')
