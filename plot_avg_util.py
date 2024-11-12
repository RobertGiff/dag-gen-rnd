import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Base directory path where data is stored
data_dir = 'data'

# Regex pattern to extract utilization from directory name
dir_pattern = re.compile(r'data-multi-m4-u([0-9\.]+)')

# Dictionary to store utilizations for each directory
util_dict = {}
all_utils = {}  # To store each individual taskset utilization

# Traverse through the data directory
for subdir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(subdir_path):
        match = dir_pattern.match(subdir)
        if match:
            util_value = float(match.group(1))
            total_utils = []

            # Traverse through the taskset directories in the subdirectory
            for taskset_dir in os.listdir(subdir_path):
                taskset_path = os.path.join(subdir_path, taskset_dir)
                if os.path.isdir(taskset_path):
                    # Read utilization values from the files
                    utilization = []
                    for file in os.listdir(taskset_path):
                        if file.startswith('Tau_') and file.endswith('.gml'):
                            file_path = os.path.join(taskset_path, file)
                            with open(file_path, 'r') as f:
                                for line in f:
                                    if 'U ' in line:
                                        utilization.append(float(line.split()[1]))
                    # Append the total utilization of the current taskset
                    if utilization:
                        total_util = sum(utilization)
                        total_utils.append(total_util)
                        if util_value not in all_utils:
                            all_utils[util_value] = []
                        all_utils[util_value].append(total_util)

            # Calculate average of total utilizations for the current subdir
            if total_utils:
                avg_util = np.mean(total_utils)
                util_dict[util_value] = avg_util

# Sort the utilization dictionary by the x-axis values (utilization)
x_vals = sorted(util_dict.keys())
y_vals = [util_dict[x] for x in x_vals]

# Plotting the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# First plot: Average sum utilization
ax1.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Average Sum Utilization')
ax1.plot(x_vals, x_vals, linestyle='--', color='r', label='y = x (Ideal Reference)')
ax1.set_xlabel('Utilization of Taskset (uX)')
ax1.set_ylabel('Average Sum Total Utilization')
ax1.set_title('Average Sum Utilization of Tasksets')
ax1.grid(True)
ax1.legend()

# Second plot: Individual utilizations without averaging
for util_value in sorted(all_utils.keys()):
    ax2.scatter([util_value] * len(all_utils[util_value]), all_utils[util_value], color='b', alpha=0.6)
ax2.plot(x_vals, x_vals, linestyle='--', color='r', label='y = x (Ideal Reference)')
ax2.set_xlabel('Utilization of Taskset (uX)')
ax2.set_ylabel('Individual Total Utilizations')
ax2.set_title('Individual Taskset Utilizations')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
