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

# Plotting the box plots of the time taken for torch default
# Text file will contain the following line structures: 
# Size:  16
# Time taken for im2COL: 0.000702404 s
# Time taken for whole default 0.0013697147369384766
file_path = '../../plots/relative_increase_default.txt'

sizes = []
im2col_times = []
default_times = []

with open(file_path, 'r') as file:
    size = None
    im2col_time = None
    default_time = None

    for line in file:
        if "Time taken for window to columns" in line:
            im2col_time = float(re.findall(r'[\d.]+', line)[0])
        elif "Size:" in line:
            size = int(re.findall(r'\d{2,}', line)[0])
        elif "Time taken for whole default" in line:
            default_time = float(re.findall(r'[\d.]+', line)[0])
        
        # Append data only when all three values are captured
        if size is not None and im2col_time is not None and default_time is not None:
            sizes.append(size)
            im2col_times.append(im2col_time)
            default_times.append(default_time)
            size = None
            im2col_time = None
            default_time = None

# Create a DataFrame
data = {
    'Size': sizes,
    'Time taken for im2COL': im2col_times,
    'Time taken for whole default': default_times
}
df = pd.DataFrame(data)

# Calculate mean times for each size
mean_times = df.groupby('Size').mean().reset_index()
print(mean_times)

# Plotting the average times
fig, ax = plt.subplots(figsize=(20, 8))
fig.suptitle('Average Time Taken for Different Functions by Input Size')

# Plotting the average times for both functions
width = 0.35  # the width of the bars
indices = range(len(mean_times['Size']))  # the x locations for the groups

# Plotting each set of data with an offset (width) to distinguish them
ax.bar(indices, mean_times['Time taken for im2COL'], width, label='im2COL')
ax.bar([i + width for i in indices], mean_times['Time taken for whole default'], width, label='Whole Default')

ax.set_xlabel('Size')
ax.set_ylabel('Average Time (s)')
ax.set_title('Average Time by Size and Function')
ax.set_xticks([i + width / 2 for i in indices])
ax.set_xticklabels(mean_times['Size'])
ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("../../plots/average_time_functions_by_size_default.png", bbox_inches='tight')
plt.show()