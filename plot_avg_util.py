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
                        total_utils.append(sum(utilization))

            # Calculate average of total utilizations for the current subdir
            if total_utils:
                avg_util = np.mean(total_utils)
                util_dict[util_value] = avg_util

# Sort the utilization dictionary by the x-axis values (utilization)
x_vals = sorted(util_dict.keys())
print(f"x_vals: {x_vals}")
y_vals = [util_dict[x] for x in x_vals]
print(f"y_vals: {y_vals}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Average Sum Utilization')
plt.plot(x_vals, x_vals, linestyle='--', color='r', label='y = x (Ideal Reference)')
plt.xlabel('Utilization of Taskset (uX)')
plt.ylabel('Average Sum Total Utilization')
plt.title('Average Sum Utilization of Tasksets')
plt.grid(True)
plt.legend()
plt.show()