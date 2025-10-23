"""
Llama 3.2 3B Instruct Mystery Plot

Shows the full range (-5 to +5) of steering data for Llama 3.2 3B
to visualize the U-shape pathology that makes this model an outlier.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration (easy to edit!)
MODEL_PATH = Path('models/llama-3.2-3b-instruct/output')
CSV_FILE = 'steering_quantitative_results.csv'
ALPHA_MIN = -5.0
ALPHA_MAX = 5.0

# Plot styling
PLOT_COLOR = '#9467bd'
MARKER_SIZE = 100
MARKER_ALPHA = 0.7
FIGURE_SIZE = (12, 8)

# Linear fit ranges (easy to adjust!)
LEFT_FIT_MIN = -2.5
LEFT_FIT_MAX = 0.0
RIGHT_FIT_MIN = 0.5
RIGHT_FIT_MAX = 3.0

# Output
OUTPUT_FILE = Path('llama_3b_mystery.png')

# Load data
csv_path = MODEL_PATH / CSV_FILE
df = pd.read_csv(csv_path)

# Filter to desired range
df_plot = df[
    (df['steering_strength'] >= ALPHA_MIN) &
    (df['steering_strength'] <= ALPHA_MAX)
].copy()

print(f"Loaded {len(df_plot)} data points from α={ALPHA_MIN} to α={ALPHA_MAX}")

# Compute linear fits for two regions
# Left region (negative α)
df_left = df_plot[
    (df_plot['steering_strength'] >= LEFT_FIT_MIN) &
    (df_plot['steering_strength'] <= LEFT_FIT_MAX)
].copy()

slope_left, intercept_left, r_left, p_left, _ = stats.linregress(
    df_left['steering_strength'],
    df_left['flesch_kincaid_grade']
)
r2_left = r_left ** 2

# Right region (positive α)
df_right = df_plot[
    (df_plot['steering_strength'] >= RIGHT_FIT_MIN) &
    (df_plot['steering_strength'] <= RIGHT_FIT_MAX)
].copy()

slope_right, intercept_right, r_right, p_right, _ = stats.linregress(
    df_right['steering_strength'],
    df_right['flesch_kincaid_grade']
)
r2_right = r_right ** 2

print(f"\nLeft fit (α ∈ [{LEFT_FIT_MIN}, {LEFT_FIT_MAX}]): R²={r2_left:.3f}, slope={slope_left:.3f}")
print(f"Right fit (α ∈ [{RIGHT_FIT_MIN}, {RIGHT_FIT_MAX}]): R²={r2_right:.3f}, slope={slope_right:.3f}")

# Create plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# Scatterplot
ax.scatter(df_plot['steering_strength'],
           df_plot['flesch_kincaid_grade'],
           s=MARKER_SIZE,
           alpha=MARKER_ALPHA,
           color=PLOT_COLOR,
           edgecolors='white',
           linewidth=2,
           zorder=3)

# Plot linear fit lines
# Left fit
x_left = np.array([LEFT_FIT_MIN, LEFT_FIT_MAX])
y_left = slope_left * x_left + intercept_left
ax.plot(x_left, y_left,
        color='#d62728', linestyle='--', linewidth=3, alpha=0.8,
        label=f'Left fit: y={slope_left:.2f}x+{intercept_left:.1f} (R²={r2_left:.3f})',
        zorder=2)

# Right fit
x_right = np.array([RIGHT_FIT_MIN, RIGHT_FIT_MAX])
y_right = slope_right * x_right + intercept_right
ax.plot(x_right, y_right,
        color='#2ca02c', linestyle='--', linewidth=3, alpha=0.8,
        label=f'Right fit: y={slope_right:.2f}x+{intercept_right:.1f} (R²={r2_right:.3f})',
        zorder=2)

# Baseline reference
ax.axvline(0, color='gray', linestyle=':', alpha=0.4, linewidth=2,
           label='No steering (α=0)', zorder=1)

# Labels and styling
ax.set_xlabel('Steering Strength (α)', fontsize=13, fontweight='bold')
ax.set_ylabel('Flesch-Kincaid Grade Level', fontsize=13, fontweight='bold')
ax.set_title('Llama 3.2 3B Instruct: The Outlier Mystery\n(Full steering range: α ∈ [-5, +5])',
             fontsize=15, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

# Legend
ax.legend(loc='best', fontsize=11, framealpha=0.95)

plt.tight_layout()

# Save
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved mystery plot to {OUTPUT_FILE}")

plt.close()

# Print some stats
print(f"\nGrade level range: {df_plot['flesch_kincaid_grade'].min():.1f} to {df_plot['flesch_kincaid_grade'].max():.1f}")
print(f"At α=0: {df_plot[df_plot['steering_strength'].abs() < 0.01]['flesch_kincaid_grade'].mean():.1f}")
