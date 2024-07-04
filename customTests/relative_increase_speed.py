import pandas as pd
import matplotlib.pyplot as plt
import re

# Read the data from the file
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

# Calculate the mean times for each size
mean_times = df.groupby('Size').mean().reset_index()
# printing mean times
print(mean_times)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each time category
categories = ['Time to identify windows', 'Time to copy windows', 'Time to perform sgemm_', 'Time to re-insert NaNs']
for category in categories:
    ax.plot(mean_times['Size'], mean_times[category], label=category)

ax.set_xlabel('Size')
ax.set_ylabel('Time (s)')
ax.set_title('Average Time Taken for Different Functions by Input Size')
ax.legend()

# Add the DataFrame as text below the plot
text_str = mean_times.to_string(index=False, col_space=40)
plt.figtext(0.5, -0.1, text_str, ha="center", fontsize=9, va="top")

plt.tight_layout()

# Save the plot
plt.savefig("../../plots/average_time_functions_by_size.png", bbox_inches='tight')
plt.show()
