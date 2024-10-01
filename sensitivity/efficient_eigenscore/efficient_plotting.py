import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have the data in a pandas DataFrame

df = pd.read_csv('ablation_study_results.csv')

# Creating the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotting EES Time
ax.plot(df['Rows'], df['Cols'], df['EES Time'], label='EES Time', marker='o', color='blue')

# Plotting Regular Time
ax.plot(df['Rows'], df['Cols'], df['Regular Time'], label='Regular Time', marker='o', color='red')

# Setting labels
ax.set_xlabel('Rows')
ax.set_ylabel('Cols')
ax.set_zlabel('Time (s)')
ax.set_title('Comparison of EES Time and Regular Time')

# Adjusting the view angle for better visibility
ax.view_init(elev=10, azim=140)  # Lower elev rotates downward, higher azim rotates eastward

# Adding a legend
ax.legend()

# Saving the plot to a file (e.g., PNG format)
plt.savefig('comparison_plot_rotated.png')
