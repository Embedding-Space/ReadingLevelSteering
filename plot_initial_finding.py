"""
Simple scatterplot showing the initial linear relationship discovery.

Creates a clean visualization of steering strength vs. Flesch-Kincaid grade level
with regression line and statistics to illustrate the "there's something here"
moment that justified further investigation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

# Load data
data_path = Path("./output/steering_quantitative_results.csv")
df = pd.read_csv(data_path)

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df['steering_strength'],
    df['flesch_kincaid_grade']
)
r_squared = r_value ** 2

# Create regression line
x_line = np.array([df['steering_strength'].min(), df['steering_strength'].max()])
y_line = slope * x_line + intercept

# Create plot
fig, ax = plt.subplots(figsize=(10, 7))

# Scatterplot
ax.scatter(df['steering_strength'], df['flesch_kincaid_grade'],
           s=80, alpha=0.6, color='#2E86AB', edgecolors='white', linewidth=1.5,
           zorder=3)

# Regression line
ax.plot(x_line, y_line,
        color='#A23B72', linewidth=2.5, linestyle='--',
        label=f'Linear fit (R² = {r_squared:.3f})',
        zorder=2)

# Baseline reference
ax.axvline(0, color='gray', linestyle=':', alpha=0.4, linewidth=1.5,
           label='No steering', zorder=1)

# Labels and title
ax.set_xlabel('Steering Strength', fontsize=13, fontweight='bold')
ax.set_ylabel('Flesch-Kincaid Grade Level', fontsize=13, fontweight='bold')
ax.set_title('Initial Discovery: Steering Affects Reading Complexity',
             fontsize=15, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Add stats annotation
stats_text = f'R² = {r_squared:.3f}\np-value = {p_value:.2e}\nslope = {slope:.2f}'
ax.text(0.05, 0.05, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save
output_path = Path("./output/initial_linear_finding.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to {output_path}")

# Print stats for reference
print(f"\nRegression Statistics:")
print(f"  R² = {r_squared:.4f}")
print(f"  p-value = {p_value:.2e}")
print(f"  slope = {slope:.4f}")
print(f"  intercept = {intercept:.4f}")

plt.close()
