import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Open file for writing
choice = int(os.getenv('DEFAULT', 0))
if choice == 0:
    file_path = '../../plots/relative_increase.txt'

    sizes = []
    identify_times = []
    copy_times = []
    sgemm_times = []
    reinsert_nans_times = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Size:" in line:
                size = int(re.findall(r'\d+', line)[0])
                sizes.append(size)
            elif "Time taken to identify windows" in line:
                identify_times.append(float(re.findall(r'[\d.]+', line)[0]))
            elif "Time taken to copy windows" in line:
                copy_times.append(float(re.findall(r'[\d.]+', line)[0]))
            elif "Time taken to perform sgemm_" in line:
                sgemm_times.append(float(re.findall(r'[\d.]+', line)[0]))
            elif "Time taken to re-insert NaNs" in line:
                reinsert_nans_times.append(float(re.findall(r'[\d.]+', line)[0]))

    # Create a DataFrame
    data = {
        'Size': sizes,
        'Time to identify windows': identify_times,
        'Time to copy windows': copy_times,
        'Time to perform sgemm_': sgemm_times,
        'Time to re-insert NaNs': reinsert_nans_times
    }
    df = pd.DataFrame(data)

    # Get unique sizes
    unique_sizes = df['Size'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharey=True)
    fig.suptitle('Time Taken for Different Functions by Input Size')

    axes = axes.flatten()

    for ax, size in zip(axes, unique_sizes):
        subset = df[df['Size'] == size]
        subset.boxplot(column=['Time to identify windows', 'Time to copy windows', 'Time to perform sgemm_', 'Time to re-insert NaNs'], ax=ax, grid=False)
        ax.set_title(f'Size {size}')
        ax.set_xlabel('Function')
        ax.set_ylabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../../plots/boxplot_time_functions_by_size.png", bbox_inches='tight')
    plt.show()

elif choice == 1:
    file_path = '../../plots/relative_increase_windows.txt'

    sizes = []
    identify_times = []
    copy_times = []
    sgemm_times = []
    reinsert_nans_times = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Size:" in line:
                size = int(re.findall(r'\d+', line)[0])
                sizes.append(size)
            elif "Time taken to identify windows" in line:
                identify_times.append(float(re.findall(r'[\d.]+', line)[0]))
            elif "Time taken to copy windows" in line:
                copy_times.append(float(re.findall(r'[\d.]+', line)[0]))
            elif "Time taken to perform sgemm_" in line:
                sgemm_times.append(float(re.findall(r'[\d.]+', line)[0]))
            elif "Time taken to re-insert NaNs" in line:
                reinsert_nans_times.append(float(re.findall(r'[\d.]+', line)[0]))

    # Create a DataFrame
    data = {
        'Size': sizes,
        'Time to identify windows': identify_times,
        'Time to copy windows': copy_times,
        'Time to perform sgemm_': sgemm_times,
        'Time to re-insert NaNs': reinsert_nans_times
    }
    df = pd.DataFrame(data)

    # Get unique sizes
    unique_sizes = df['Size'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharey=True)
    fig.suptitle('Time Taken for Different Functions by Input Size')

    axes = axes.flatten()

    for ax, size in zip(axes, unique_sizes):
        subset = df[df['Size'] == size]
        subset.boxplot(column=['Time to identify windows', 'Time to copy windows', 'Time to perform sgemm_', 'Time to re-insert NaNs'], ax=ax, grid=False)
        ax.set_title(f'Size {size}')
        ax.set_xlabel('Function')
        ax.set_ylabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../../plots/boxplot_time_functions_by_size.png", bbox_inches='tight')
    plt.show()
