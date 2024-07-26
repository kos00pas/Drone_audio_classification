import os

def find_file_in_subdirectories(root_dir, target_filename):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            print(f"File found: {os.path.join(dirpath, target_filename)}")
            return os.path.join(dirpath, target_filename)
    print("File not found.")
    return None

if __name__ == "__main__":
    root_directory = '.'  # Change this to the directory you want to start the search from
    target_file = 'summary_results.txt'

    file_path = find_file_in_subdirectories(root_directory, target_file)

    if file_path:
        print(f"Summary file is located at: {file_path}")
    else:
        print("Summary file not found in any subdirectories.")
