import csv

def extract_paths_and_process(file_path):
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if there is one
        for row in csv_reader:
            modified_folder = row[0]  # Modify the folder data_mp3
            shape_str = row[1].strip('()')
            if shape_str:  # Check if shape_str is not empty
                try:
                    shape = tuple(map(int, shape_str.split(',')))  # Convert shape to tuple of integers
                    if shape[0] < 16000:
                        # print(f"Path: {modified_folder}, Shape: {shape}")
                        padd(modified_folder)
                except ValueError:
                    print(f"Skipping invalid shape: {shape_str}")

def padd(path):
    target_shape = 16000
    try:
        with open(path, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            data_exist = []
            for i, row in enumerate(csv_reader):
                if i == 0:  # Skip the header row
                    continue
                data_exist.append(row)

            # Get the shape of data_exist
            num_rows = len(data_exist)
            num_cols = len(data_exist[0]) if num_rows > 0 else 0
            data_shape = (num_rows, num_cols)
            print("Shape of data_exist:", data_shape)

            # Initialize the output data_mp3 with the existing data_mp3
            output_data = data_exist.copy()

            # Pad the data_mp3 until it reaches the target shape of 16000 rows
            while len(output_data) < target_shape:
                remaining_rows = target_shape - len(output_data)
                output_data.extend(data_exist[:remaining_rows])

            # If the length exceeds the target shape, trim it
            output_data = output_data[:target_shape]

            print("Padded Data Shape:", (len(output_data), num_cols),path)

        # Write the padded data_mp3 back to the file
        with open(path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Column1', 'Column2'])  # Write header row if needed
            csv_writer.writerows(output_data)

    except FileNotFoundError:
        print(f"File not found: {path}")

file_path = 'extracted_info.csv'
extract_paths_and_process(file_path)
