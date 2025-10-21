#!/usr/bin/env python3
"""
Meta-analysis: Prompted Control Across All Models

Visualizes how well models can follow explicit grade-level instructions
by plotting actual FK grade vs requested grade for all tested models.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# Models with prompted control data
models = [
    ("qwen3-0.6b", "#E63946"),           # Red
    ("qwen3-1.7b", "#F77F00"),          # Orange
    ("qwen3-4b-instruct", "#FCBF49"),   # Yellow/Gold
    ("llama-3.2-1b-instruct", "#06A77D"), # Teal
    ("gemma-3-4b-it", "#4EA8DE"),       # Blue
    ("Phi-3-mini-4k-instruct", "#5F0F40"), # Purple
    ("granite-4.0-micro", "#9D4EDD"),   # Lavender
]

# Collect all data
all_data = []
stats = []

for model_name, color in models:
    csv_path = Path(f"models/{model_name}/output/prompted_control_results.csv")
    if not csv_path.exists():
        print(f"⚠️  Skipping {model_name}: no data")
        continue

    df = pd.read_csv(csv_path)
    df['model'] = model_name
    df['color'] = color
    all_data.append(df)

    # Calculate R² for this model
    slope, intercept, r_value, _, _ = linregress(
        df['requested_grade'],
        df['actual_grade']
    )
    r_squared = r_value ** 2

    stats.append({
        'model': model_name,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept,
        'color': color,
        'mean_actual': df['actual_grade'].mean(),
        'std_actual': df['actual_grade'].std(),
    })

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)
stats_df = pd.DataFrame(stats)

print("\n" + "="*80)
print("PROMPTED CONTROL META-ANALYSIS")
print("="*80)
print(f"\n{'Model':<30} {'R²':>8} {'Mean FK':>10} {'Std FK':>10} {'Slope':>8}")
print("-"*80)
for _, row in stats_df.iterrows():
    print(f"{row['model']:<30} {row['r_squared']:>8.4f} {row['mean_actual']:>10.2f} "
          f"{row['std_actual']:>10.2f} {row['slope']:>8.3f}")
print("="*80 + "\n")

# Create visualization
fig, ax = plt.subplots(figsize=(12, 10))

# Plot each model's data points
for model_name, color in models:
    model_data = combined_df[combined_df['model'] == model_name]
    if len(model_data) == 0:
        continue

    # Get R² for label
    model_stats = stats_df[stats_df['model'] == model_name].iloc[0]
    r_sq = model_stats['r_squared']

    ax.scatter(
        model_data['requested_grade'],
        model_data['actual_grade'],
        color=color,
        s=60,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        label=f"{model_name} (R²={r_sq:.3f})"
    )

# Add ideal 1:1 line (where points SHOULD fall)
requested_range = [5, 20]
ax.plot(
    requested_range,
    requested_range,
    'gray',
    linestyle=':',
    linewidth=2.5,
    label='Ideal response (slope=1.0)',
    alpha=0.7,
    zorder=1
)

# Styling
ax.set_xlabel('Requested Grade Level (via prompt)', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual Flesch-Kincaid Grade', fontsize=13, fontweight='bold')
ax.set_title(
    'Prompted Control Failure: Models Ignore Grade-Level Instructions\n'
    'All models cluster around FK~7-8 regardless of requested complexity',
    fontsize=14,
    fontweight='bold',
    pad=20
)
ax.grid(True, alpha=0.3, zorder=0)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

# Set axis limits with some padding
ax.set_xlim(4, 21)
ax.set_ylim(4, 21)

# Add diagonal reference lines for context
for grade in [5, 10, 15, 20]:
    ax.axhline(y=grade, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)
    ax.axvline(x=grade, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)

# Add annotation box explaining the failure
textstr = (
    'Expected: Points along diagonal (model obeys instructions)\n'
    'Observed: Horizontal cloud at FK≈7-8 (model ignores instructions)\n\n'
    'Conclusion: Prompting does NOT provide reliable reading-level control'
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.text(
    0.98, 0.28,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=props
)

plt.tight_layout()

# Save
output_path = Path("prompted_control_meta_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}\n")

# Calculate overall statistics
overall_r_sq_mean = stats_df['r_squared'].mean()
overall_r_sq_std = stats_df['r_squared'].std()

print(f"📊 OVERALL STATISTICS:")
print(f"   Mean R² across models: {overall_r_sq_mean:.4f} ± {overall_r_sq_std:.4f}")
print(f"   Range of R²: {stats_df['r_squared'].min():.4f} to {stats_df['r_squared'].max():.4f}")
print(f"\n   Compare to steering R² (qwen3-4b): 0.8963")
print(f"   Prompting is {(0.8963 / overall_r_sq_mean):.1f}x worse at controlling reading level\n")
