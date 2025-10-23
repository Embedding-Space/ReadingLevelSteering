"""
Compile Linear Regression Statistics Across All Models

Reads quantify_steering results from each model directory and computes
regression statistics for grade level vs steering strength.

Outputs a CSV table suitable for import into Google Sheets/Docs.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Model metadata
MODELS = {
    'qwen3-0.6b': {
        'name': 'Qwen 3 0.6B',
        'params': '625M',
        'path': 'models/qwen3-0.6b/output'
    },
    'qwen3-1.7b': {
        'name': 'Qwen 3 1.7B',
        'params': '2.0B',
        'path': 'models/qwen3-1.7b/output'
    },
    'qwen3-4b-instruct': {
        'name': 'Qwen 3 4B Instruct',
        'params': '4.3B',
        'path': 'models/qwen3-4b-instruct/output'
    },
    'llama-3.2-1b-instruct': {
        'name': 'Llama 3.2 1B Instruct',
        'params': '1.2B',
        'path': 'models/llama-3.2-1b-instruct/output'
    },
    'llama-3.2-3b-instruct': {
        'name': 'Llama 3.2 3B Instruct',
        'params': '3.2B',
        'path': 'models/llama-3.2-3b-instruct/output'
    },
    'gemma-3-4b-it': {
        'name': 'Gemma 3 4B IT',
        'params': '4.0B',
        'path': 'models/gemma-3-4b-it/output'
    },
    'Phi-3-mini-4k-instruct': {
        'name': 'Phi-3 Mini 4K Instruct',
        'params': '3.8B',
        'path': 'models/Phi-3-mini-4k-instruct/output'
    },
    'granite-4.0-micro': {
        'name': 'Granite 4.0 Micro',
        'params': '400M',
        'path': 'models/granite-4.0-micro/output'
    },
}

# Effective range for filtering (based on documented research)
EFFECTIVE_ALPHA_MIN = -4.0
EFFECTIVE_ALPHA_MAX = 4.0

results = []

print("Compiling regression statistics across all models...\n")

for model_id, model_info in MODELS.items():
    model_path = Path(model_info['path'])
    csv_path = model_path / 'steering_quantitative_results.csv'

    if not csv_path.exists():
        print(f"⚠ Skipping {model_info['name']}: No results file found at {csv_path}")
        continue

    # Load data
    df = pd.read_csv(csv_path)

    # Filter to effective range
    df_effective = df[
        (df['steering_strength'] >= EFFECTIVE_ALPHA_MIN) &
        (df['steering_strength'] <= EFFECTIVE_ALPHA_MAX)
    ].copy()

    if len(df_effective) < 3:
        print(f"⚠ Skipping {model_info['name']}: Insufficient data points in effective range")
        continue

    # Compute linear regression: FK_grade = slope * alpha + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_effective['steering_strength'],
        df_effective['flesch_kincaid_grade']
    )

    r_squared = r_value ** 2

    # Determine effective grade range
    min_alpha = df_effective['steering_strength'].min()
    max_alpha = df_effective['steering_strength'].max()
    min_grade = slope * min_alpha + intercept
    max_grade = slope * max_alpha + intercept
    grade_span = abs(max_grade - min_grade)

    # Format p-value with appropriate precision
    if p_value < 1e-15:
        p_formatted = f"<1×10⁻¹⁵"
    elif p_value < 0.001:
        # Scientific notation
        exponent = int(np.floor(np.log10(p_value)))
        mantissa = p_value / (10 ** exponent)
        p_formatted = f"{mantissa:.2f}×10^{exponent}"
    else:
        p_formatted = f"{p_value:.4f}"

    results.append({
        'Model': model_info['name'],
        'Parameters': model_info['params'],
        'R²': f"{r_squared:.4f}",
        'p-value': p_formatted,
        'Slope': f"{slope:.3f}",
        'Intercept': f"{intercept:.2f}",
        'Grade Span': f"{grade_span:.1f}",
        'n': len(df_effective)
    })

    print(f"✓ {model_info['name']}: R²={r_squared:.4f}, p={p_formatted}, slope={slope:.3f}")

# Create DataFrame and sort by parameter count
df_results = pd.DataFrame(results)

# Custom sort by parameter count (convert to numeric)
def param_to_number(param_str):
    if 'M' in param_str:
        return float(param_str.replace('M', '')) / 1000  # Convert to billions
    elif 'B' in param_str:
        return float(param_str.replace('B', ''))
    return 0

df_results['_sort_key'] = df_results['Parameters'].apply(param_to_number)
df_results = df_results.sort_values('_sort_key').drop('_sort_key', axis=1)

# Save to CSV
output_path = Path('model_regression_stats.csv')
df_results.to_csv(output_path, index=False)

print(f"\n{'='*80}")
print("SUMMARY TABLE")
print('='*80)
print(df_results.to_string(index=False))

print(f"\n✓ Saved to {output_path}")
print("\nYou can import this CSV directly into Google Sheets!")
