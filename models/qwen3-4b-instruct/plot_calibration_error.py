"""
Calibration Error Visualization

Creates a residual plot showing prediction errors across target grade levels.
Visualizes the ±Q factor: "Request grade P, get P ± Q"

Uses existing calibration_results.csv data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
DATA_PATH = Path("./output/calibration_results.csv")
OUTPUT_PATH = Path("./output/calibration_error.png")

# Load data
df = pd.read_csv(DATA_PATH)

# Filter to effective range
df_effective = df[df['target_grade'] <= 16].copy()

# Compute error statistics
mae = np.mean(np.abs(df_effective['error']))
rmse = np.sqrt(np.mean(df_effective['error']**2))
max_error = np.max(np.abs(df_effective['error']))

print(f"Error Statistics (grades 6-16):")
print(f"  MAE: {mae:.2f} grades")
print(f"  RMSE: {rmse:.2f} grades")
print(f"  Max absolute error: {max_error:.2f} grades")
print(f"\nSummary: Request grade P → get P ± {mae:.1f} grades (typical)")

# Create residual plot
fig, ax = plt.subplots(figsize=(12, 7))

# Shaded error band (±MAE)
ax.axhspan(-mae, mae, alpha=0.2, color='#6A994E',
           label=f'±MAE ({mae:.2f} grades)', zorder=1)

# Perfect calibration line (zero error)
ax.axhline(0, color='black', linestyle='--', linewidth=2,
           alpha=0.5, label='Perfect calibration', zorder=2)

# Scatter plot of errors
ax.scatter(df_effective['target_grade'], df_effective['error'],
           s=120, alpha=0.7, color='#2E86AB', edgecolors='white', linewidth=2,
           label='Measured error', zorder=3)

# Labels and styling
ax.set_xlabel('Target Grade Level', fontsize=13, fontweight='bold')
ax.set_ylabel('Prediction Error (Actual - Target) [grades]', fontsize=13, fontweight='bold')
ax.set_title('Grade Level Prediction Accuracy\n(How close do we get to the requested reading level?)',
             fontsize=15, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='both')

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Stats box
stats_text = f'Typical error: ±{mae:.2f} grades\nWorst case: ±{max_error:.2f} grades\n\n"Request grade P,\n get P ± {mae:.1f}"'
ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Set y-axis limits to show the error range nicely
y_margin = 0.5
ax.set_ylim(-max_error - y_margin, max_error + y_margin)

# Set x-axis limits
ax.set_xlim(5.5, 16.5)

plt.tight_layout()

# Save
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved error plot to {OUTPUT_PATH}")

plt.close()
