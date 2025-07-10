import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from datetime import datetime


# From the temp_log_20250509_180121.txt, filter out essential content for plots using this below command, to produce filtered.txt
# !awk '/^Time:/ {time=$0} /Elapsed:/ {elapsed=$2} /"cpu-thermal"/ {gsub(/[{}"]/,""); split($0,a,","); print time ", Elapsed: " elapsed ", " a[1] ", " a[2]}' temp_log_20250509_180121.txt  > filtered.txt

#

with open('/content/filtered.txt', 'r') as f:
    data_string = f.read()

# Parse the data
lines = data_string.strip().split('\n')
timestamps = []
elapsed_hours = []
cpu_temps = []
soc_temps = []

pattern = r'Time: (.+?), Elapsed: (\d+)s, cpu-thermal:(\d+) C, soc-thermal:(\d+) C'

for line in lines:
   match = re.match(pattern, line)
   if match:
       timestamp_str, elapsed, cpu_temp, soc_temp = match.groups()
       elapsed_hours.append(int(elapsed) / 3600)
       cpu_temps.append(int(cpu_temp))
       soc_temps.append(int(soc_temp))

# Convert to numpy arrays
elapsed_hours = np.array(elapsed_hours)
cpu_temps = np.array(cpu_temps)
soc_temps = np.array(soc_temps)

# Calculate rolling averages (trends)
window_size = max(5, len(cpu_temps) // 20)
cpu_trend = pd.Series(cpu_temps).rolling(window=window_size, center=True).mean()
soc_trend = pd.Series(soc_temps).rolling(window=window_size, center=True).mean()

# Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Plot raw temperature data
ax.plot(elapsed_hours, cpu_temps, color='#3498db', linewidth=1, alpha=0.8, label='CPU Temperature')
ax.plot(elapsed_hours, soc_temps, color='#e74c3c', linewidth=1, alpha=0.8, label='SoC Temperature')

# Plot trend lines
ax.plot(elapsed_hours, cpu_trend, color='#2980b9', linewidth=3, alpha=0.6, linestyle='--', label='CPU Trend')
ax.plot(elapsed_hours, soc_trend, color='#c0392b', linewidth=3, alpha=0.6, linestyle='--', label='SoC Trend')

# Customize the plot
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title('Temperature Time Series with Trends', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Set y-axis limits with some padding
temp_min = min(cpu_temps.min(), soc_temps.min()) - 1
temp_max = max(cpu_temps.max(), soc_temps.max()) + 1
ax.set_ylim(temp_min, temp_max)

# Improve layout
plt.tight_layout()

# Save the plot
plt.savefig('thermal_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print(f"Plot saved as 'thermal_plot.png'")
print(f"Data points: {len(cpu_temps)}")
print(f"Duration: {elapsed_hours.max():.1f} hours")
print(f"CPU range: {cpu_temps.min()}°C - {cpu_temps.max()}°C")
print(f"SoC range: {soc_temps.min()}°C - {soc_temps.max()}°C")
