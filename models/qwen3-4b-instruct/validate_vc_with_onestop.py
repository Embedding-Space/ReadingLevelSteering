#!/usr/bin/env python3
"""
Validate V_c by testing internal model representations against OneStop dataset.

Hypothesis: If V_c captures the model's internal understanding of reading level,
then (activation · V_c) should correlate linearly with actual FK grade level.

This proves V_c is semantically meaningful to the model, not just an extraction artifact.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textstat
from scipy.stats import linregress
import json

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
STEERING_LAYER = 35

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent
ONESTOP_PATH = SCRIPT_DIR / "../../data/onestop_pairs/onestop_pairs.json"
VC_PATH = SCRIPT_DIR / "output/complexity_vectors.pt"
OUTPUT_DIR = SCRIPT_DIR / "output"


def load_onestop_data():
    """Load and prepare OneStop articles."""
    print(f"\n{'='*80}")
    print("LOADING ONESTOP DATASET")
    print(f"{'='*80}\n")

    if not ONESTOP_PATH.exists():
        raise FileNotFoundError(f"OneStop data not found at {ONESTOP_PATH}")

    with open(ONESTOP_PATH) as f:
        data = json.load(f)

    # Extract pairs (simple and regular texts)
    texts = []
    for i, pair in enumerate(data['pairs']):
        # Simple version
        texts.append({
            'article_id': f"article_{i+1}",
            'level': 'Simple',
            'text': pair['simple_text'],
            'fk_grade': pair['simple_grade'],
            'word_count': len(pair['simple_text'].split())
        })

        # Regular version
        texts.append({
            'article_id': f"article_{i+1}",
            'level': 'Regular',
            'text': pair['regular_text'],
            'fk_grade': pair['regular_grade'],
            'word_count': len(pair['regular_text'].split())
        })

    df = pd.DataFrame(texts)

    print(f"Loaded {len(df)} texts across {df['article_id'].nunique()} articles")
    print(f"\nReading level distribution:")
    print(df.groupby('level')['fk_grade'].describe())

    return df


def get_activation_at_layer(model, tokenizer, text, layer_idx):
    """Get activation vector at specified layer after processing text."""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    # Storage for activations
    activation = None

    def hook_fn(module, input, output):
        nonlocal activation
        # Get the final token's hidden state (last position)
        # Output is already the hidden states tensor: (batch, seq_len, hidden_dim)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        activation = hidden_states[:, -1, :].detach()  # Take last token

    # Register hook
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Cleanup
    handle.remove()

    return activation.cpu().float().numpy().flatten()


def main():
    # Load OneStop data
    df = load_onestop_data()

    # Load V_c
    print(f"\n{'='*80}")
    print("LOADING COMPLEXITY VECTOR")
    print(f"{'='*80}\n")

    vc_data = torch.load(VC_PATH)
    v_c = vc_data[STEERING_LAYER].float().cpu().numpy().flatten()

    print(f"V_c loaded from layer {STEERING_LAYER}")
    print(f"Dimensionality: {len(v_c)}")
    print(f"L2 norm: {np.linalg.norm(v_c):.4f}")

    # Load model
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()

    print(f"Model loaded on {DEVICE}")

    # Compute dot products
    print(f"\n{'='*80}")
    print("COMPUTING INTERNAL REPRESENTATIONS")
    print(f"{'='*80}\n")

    results = []

    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] Processing {row['article_id']} ({row['level']})...", end=" ", flush=True)

        # Get activation
        activation = get_activation_at_layer(model, tokenizer, row['text'], STEERING_LAYER)

        # Dot product with V_c
        dot_product = np.dot(activation, v_c)

        print(f"FK={row['fk_grade']:.1f}, dot={dot_product:.2f}")

        results.append({
            'article_id': row['article_id'],
            'level': row['level'],
            'fk_grade': row['fk_grade'],
            'dot_product': dot_product,
            'word_count': row['word_count']
        })

    results_df = pd.DataFrame(results)

    # Save results
    csv_path = OUTPUT_DIR / "onestop_validation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Statistical analysis
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")

    slope, intercept, r_value, p_value, std_err = linregress(
        results_df['fk_grade'],
        results_df['dot_product']
    )
    r_squared = r_value ** 2

    print(f"Linear regression: dot_product = {slope:.4f} × FK_grade + {intercept:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"r = {r_value:.4f}")
    print(f"p-value = {p_value:.2e}")
    print(f"Standard error = {std_err:.4f}")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}\n")

    if r_squared > 0.7:
        print("✓ STRONG correlation: V_c captures model's internal reading-level representation")
        print("  → The model internally 'understands' reading level along this direction")
    elif r_squared > 0.4:
        print("✓ MODERATE correlation: V_c partially aligns with internal representation")
        print("  → Some semantic meaning, but other factors involved")
    else:
        print("✗ WEAK correlation: V_c may not reflect internal semantics")
        print("  → Could be extraction artifact or task-specific")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scatter with regression
    ax = axes[0]

    # Color by level
    level_colors = {'Simple': '#06A77D', 'Regular': '#E63946'}
    for level, color in level_colors.items():
        level_data = results_df[results_df['level'] == level]
        ax.scatter(
            level_data['fk_grade'],
            level_data['dot_product'],
            color=color,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=f"{level} (n={len(level_data)})"
        )

    # Regression line
    x_line = np.array([results_df['fk_grade'].min(), results_df['fk_grade'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Linear fit (R²={r_squared:.3f})')

    ax.set_xlabel('Actual Flesch-Kincaid Grade Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('activation · V_c (dot product)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Does the Model Internally Recognize Reading Level?\n'
        'Testing if (activation · V_c) correlates with actual FK grade',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Add stats box
    stats_text = (
        f'Slope: {slope:.3f}\n'
        f'R²: {r_squared:.4f}\n'
        f'p-value: {p_value:.2e}'
    )
    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # Plot 2: By reading level
    ax = axes[1]

    level_order = ['Simple', 'Regular']
    level_means = [results_df[results_df['level'] == lvl]['dot_product'].mean() for lvl in level_order]
    level_stds = [results_df[results_df['level'] == lvl]['dot_product'].std() for lvl in level_order]

    bars = ax.bar(
        level_order,
        level_means,
        yerr=level_stds,
        color=[level_colors[lvl] for lvl in level_order],
        edgecolor='black',
        linewidth=1.5,
        capsize=5,
        alpha=0.8
    )

    ax.set_xlabel('OneStop Reading Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean (activation · V_c)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Internal Representation by Reading Level\n'
        'Higher dot product = model "sees" higher complexity',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean FK grades as text
    for i, (lvl, mean_dot) in enumerate(zip(level_order, level_means)):
        mean_fk = results_df[results_df['level'] == lvl]['fk_grade'].mean()
        ax.text(
            i, mean_dot + level_stds[i] + 50,
            f'FK {mean_fk:.1f}',
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "onestop_validation_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {plot_path}")

    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}\n")

    if r_squared > 0.7:
        print("🎉 V_c successfully validated!")
        print("   The model internally represents reading level along this geometric direction.")
        print("   This proves V_c captures genuine semantic structure, not extraction artifact.")

    return results_df, r_squared


if __name__ == "__main__":
    results_df, r_squared = main()
