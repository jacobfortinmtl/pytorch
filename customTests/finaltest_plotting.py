import re
import os

# Get environment variable for default
default = int(os.environ.get('DEFAULT', 0))

def parse_and_average_timing(file_path):
    total_times = {
        "window_to_columns": 0.0,
        "identify_windows": 0.0,
        "copy_windows": 0.0,
        "perform_sgemm": 0.0,
        "re_insert_NaNs": 0.0,
        "convolution": 0.0
    }
    counts = {
        "window_to_columns": 0,
        "identify_windows": 0,
        "copy_windows": 0,
        "perform_sgemm": 0,
        "re_insert_NaNs": 0,
        "convolution": 0
    }
    # Open and read through the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.split(':')
            if len(parts) == 2:
                key, value = parts[0], parts[1].strip()
                # Remove 's' and convert to float
                value = "".join(re.findall(r"[0-9.]+", value))
                value = float(value)
                # Check the key and add the value to the corresponding total
                if "Time taken for window to columns" in key:
                    total_times["window_to_columns"] += value
                    counts["window_to_columns"] += 1
                elif "Time taken to identify windows" in key:
                    total_times["identify_windows"] += value
                    counts["identify_windows"] += 1
                elif "Time taken to copy windows" in key:
                    total_times["copy_windows"] += value
                    counts["copy_windows"] += 1
                elif "Time taken to perform sgemm_" in key:
                    total_times["perform_sgemm"] += value
                    counts["perform_sgemm"] += 1
                elif "Time taken to re-insert NaNs" in key:
                    total_times["re_insert_NaNs"] += value
                    counts["re_insert_NaNs"] += 1

    # Calculate and print averages
    for key in total_times:
        if counts[key] > 0:
            print(f"Average {key}: {total_times[key] / counts[key]}")
    print()

def parse_and_average_timing_default(file_path):
    total_times = {
        "perform_sgemm": 0.0
    }
    counts = {
        "perform_sgemm": 0
    }
    # Open and read through the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.split(':')
            if len(parts) == 2:
                key, value = parts[0], parts[1].strip()
                # Remove 's' and convert to float
                value = "".join(re.findall(r"[0-9.]+", value))
                value = float(value)
                # Check the key and add the value to the corresponding total
                if "Time taken to perform default sgemm_" in key:
                    total_times["perform_sgemm"] += value
                    counts["perform_sgemm"] += 1

    # Calculate and print averages
    for key in total_times:
        if counts[key] > 0:
            print(f"Average {key}: {total_times[key] / counts[key]}")
    print()
# Example usage
if default != 1:
    file_path = '../../plots/FINAL_TIMING.txt'
    parse_and_average_timing(file_path)
else:
    file_path = '../../plots/FINAL_TIMING_DEFAULT.txt'
    parse_and_average_timing_default(file_path)