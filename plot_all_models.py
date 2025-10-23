"""
Composite Plot: Grade Level vs Steering Strength Across All Models

Creates a single scatterplot showing all 8 models' data together
to visualize how different architectures respond to steering.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model metadata with colors
MODELS = {
    'qwen3-0.6b': {
        'name': 'Qwen 3 0.6B',
        'params': '625M',
        'path': 'models/qwen3-0.6b/output',
        'color': '#1f77b4',
        'marker': 'o'
    },
    'qwen3-1.7b': {
        'name': 'Qwen 3 1.7B',
        'params': '2.0B',
        'path': 'models/qwen3-1.7b/output',
        'color': '#ff7f0e',
        'marker': 's'
    },
    'qwen3-4b-instruct': {
        'name': 'Qwen 3 4B Instruct',
        'params': '4.3B',
        'path': 'models/qwen3-4b-instruct/output',
        'color': '#2ca02c',
        'marker': '^'
    },
    'llama-3.2-1b-instruct': {
        'name': 'Llama 3.2 1B',
        'params': '1.2B',
        'path': 'models/llama-3.2-1b-instruct/output',
        'color': '#d62728',
        'marker': 'D'
    },
    'llama-3.2-3b-instruct': {
        'name': 'Llama 3.2 3B (outlier)',
        'params': '3.2B',
        'path': 'models/llama-3.2-3b-instruct/output',
        'color': '#9467bd',
        'marker': 'X'
    },
    'gemma-3-4b-it': {
        'name': 'Gemma 3 4B IT',
        'params': '4.0B',
        'path': 'models/gemma-3-4b-it/output',
        'color': '#8c564b',
        'marker': 'p'
    },
    'Phi-3-mini-4k-instruct': {
        'name': 'Phi-3 Mini',
        'params': '3.8B',
        'path': 'models/Phi-3-mini-4k-instruct/output',
        'color': '#e377c2',
        'marker': 'h'
    },
    'granite-4.0-micro': {
        'name': 'Granite 4.0 Micro',
        'params': '400M',
        'path': 'models/granite-4.0-micro/output',
        'color': '#7f7f7f',
        'marker': 'v'
    },
}

# Effective range
EFFECTIVE_ALPHA_MIN = -4.0
EFFECTIVE_ALPHA_MAX = 4.0

fig, ax = plt.subplots(figsize=(14, 10))

print("Loading data from all models...\n")

for model_id, model_info in MODELS.items():
    model_path = Path(model_info['path'])
    csv_path = model_path / 'steering_quantitative_results.csv'

    if not csv_path.exists():
        print(f"⚠ Skipping {model_info['name']}: No results file")
        continue

    # Load data
    df = pd.read_csv(csv_path)

    # Filter to effective range
    df_effective = df[
        (df['steering_strength'] >= EFFECTIVE_ALPHA_MIN) &
        (df['steering_strength'] <= EFFECTIVE_ALPHA_MAX)
    ].copy()

    # Sort by alpha for clean line plot
    df_effective = df_effective.sort_values('steering_strength')

    # Plot as line
    ax.plot(df_effective['steering_strength'],
            df_effective['flesch_kincaid_grade'],
            color=model_info['color'],
            linewidth=2.5,
            alpha=0.8,
            label=f"{model_info['name']} ({model_info['params']})")

    print(f"✓ Plotted {model_info['name']}: {len(df_effective)} points")

# Baseline reference
ax.axvline(0, color='black', linestyle=':', alpha=0.3, linewidth=1.5,
           label='No steering (α=0)', zorder=1)

# Labels and styling
ax.set_xlabel('Steering Strength (α)', fontsize=13, fontweight='bold')
ax.set_ylabel('Flesch-Kincaid Grade Level', fontsize=13, fontweight='bold')
ax.set_title('Reading Level Control Across Model Architectures\n(All 8 models, effective range α ∈ [-4, +4])',
             fontsize=15, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

# Legend - put it outside to avoid covering data
ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)

plt.tight_layout()

# Save
output_path = Path("all_models_composite.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved composite plot to {output_path}")

plt.close()
