import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("all_paths_and_labels.csv")

# Count occurrences of 'drone' and 'not_drone' in the second column
drone_count = df[df.iloc[:, 1].str.contains('drone', case=False)].shape[0]
not_drone_count = df[df.iloc[:, 1].str.contains('not_drone', case=False)].shape[0]

print("Number of occurrences of 'drone':", drone_count)
print("Number of occurrences of 'not_drone':", not_drone_count)

# Create drone.csv and not_drone.csv
drone_df = df[df.iloc[:, 1].str.contains('drone', case=False)]
not_drone_df = df[df.iloc[:, 1].str.contains('not_drone', case=False)]

drone_df.iloc[:, 0].to_csv("drone.csv", index=False)
not_drone_df.iloc[:, 0].to_csv("not_drone.csv", index=False)

print("drone.csv and not_drone.csv created successfully.")
