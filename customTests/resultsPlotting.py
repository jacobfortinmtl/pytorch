import pandas as pd
import matplotlib.pyplot as plt

# Creating the DataFrame with the initial data
data_initial = {
    "Type": ["DEFAULT", "IM2COL", "IM2ROW"],
    "WINDOW IDENTIFICATION": [0, 0.000980643, 0.000887494],
    "WINDOW COPYING": [0, 0.003309536, 0.002964344],
    "SGEMM": [0.000243102, 0.000214367, 0.000352871],
    "NAN RE-INSERTION": [0, 0.001911062, 0.002001671],
    "Total": [0.000243102, 0.006415608, 0.00620638]
}

# Creating the DataFrame with the new data
data_new = {
    "Type": ["IM2COL", "IM2ROW", "Speedup factor of im2row"],
    "WINDOW IDENTIFICATION": [0.000980643, 0.000887494, 1.104957329],
    "WINDOW COPYING": [0.003309536, 0.002964344, 1.116448024],
    "SGEMM": [0.000214367, 0.000352871, 0.607493957],
    "Speedup factor for SGEMM": [1.134045819, 0.688925981, None],
    "NAN RE-INSERTION": [0.001911062, 0.002001671, 0.95473332],
    "Total": [0.006415608, 0.00620638, 1.033711761],
    "Speedup factor for total": [0.037892278, 0.039169693, None]
}

# Creating the DataFrame with the additional data
data_additional = {
    "Type": ["DEFAULT", "IM2COL", "IM2ROW"],
    "% Speedub Compared to Default": [1, 1.134, 0.6889]
}

df_initial = pd.DataFrame(data_initial)
df_new = pd.DataFrame(data_new)
df_additional = pd.DataFrame(data_additional)

# Creating the subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Plotting individual metrics for initial data
df_initial.set_index('Type').plot(kind='bar', ax=axes[0])
axes[0].set_title('Time Taken for a Specific Function')
axes[0].set_xlabel('Function')
axes[0].set_ylabel('Time (secs)')
axes[0].legend(title='Function')

# Plotting the difference between IM2COL and IM2ROW
df_diff = df_new[df_new['Type'] == "Speedup factor of im2row"].drop(columns=['NAN RE-INSERTION', 'Speedup factor for SGEMM', 'Speedup factor for total']).set_index('Type').T
df_diff.plot(kind='bar', ax=axes[1], color=['green' if x < 1 else 'red' for x in df_diff.values.flatten()])
axes[1].set_title('Speedup Factor of functions when using im2row rather than im2col window memory formatting')
axes[1].set_xlabel('Functions')
axes[1].set_ylabel('Speedup Factor')
axes[1].axhline(1, color='black', linewidth=0.5, linestyle='--')
axes[1].legend().set_visible(False)

# Plotting the additional data for SGEMM and Change VS Default
df_additional.set_index('Type').plot(kind='bar', ax=axes[2], color=['orange'])
axes[2].set_title('Speedup Factor of SGEMM for IM2COL and IM2ROW compared to DEFAULT')
axes[2].axhline(1, color='black', linewidth=0.5, linestyle='--')
axes[2].set_xlabel('Version')
axes[2].set_ylabel('Speedup Factor')
axes[2].legend().set_visible(False)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig("../../plots/final_plot_comparison.png", bbox_inches='tight')

# Show plots
plt.show()
