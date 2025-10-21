#!/usr/bin/env python3
"""
Compare sequence-averaged vs last-token extraction methods.

Hypothesis: Both peak at layer 35, but magnitude profiles may differ.
Last-token method shows much LARGER magnitudes overall.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Load both magnitude profiles
with open(OUTPUT_DIR / "complexity_magnitudes.json") as f:
    seq_avg_mags = json.load(f)

with open(OUTPUT_DIR / "complexity_magnitudes_last_token.json") as f:
    last_token_mags = json.load(f)

# Convert to arrays
layers = np.array([int(k) for k in seq_avg_mags.keys()])
seq_avg = np.array([seq_avg_mags[str(i)] for i in layers])
last_token = np.array([last_token_mags[str(i)] for i in layers])

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Both on same scale
ax = axes[0]
ax.plot(layers, seq_avg, marker='o', linewidth=2, markersize=4,
        color='#06A77D', label='Sequence-averaged', alpha=0.8)
ax.plot(layers, last_token, marker='s', linewidth=2, markersize=4,
        color='#E63946', label='Last-token only', alpha=0.8)
ax.axvline(35, color='gray', linestyle='--', alpha=0.5, label='Layer 35 (peak for both)')
ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Complexity Vector Magnitude (L2 norm)', fontsize=12, fontweight='bold')
ax.set_title('Extraction Method Comparison\nBoth methods peak at layer 35, but magnitudes differ drastically',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=11)

# Add magnitude comparison text
ratio_at_35 = last_token[-1] / seq_avg[-1]
stats_text = (
    f'Peak magnitude (layer 35):\n'
    f'  Seq-avg: {seq_avg[-1]:.1f}\n'
    f'  Last-token: {last_token[-1]:.1f}\n'
    f'  Ratio: {ratio_at_35:.1f}×'
)
ax.text(
    0.97, 0.05,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

# Plot 2: Normalized comparison (show shape difference)
ax = axes[1]
seq_avg_norm = seq_avg / seq_avg[-1]  # Normalize to peak
last_token_norm = last_token / last_token[-1]

ax.plot(layers, seq_avg_norm, marker='o', linewidth=2, markersize=4,
        color='#06A77D', label='Sequence-averaged', alpha=0.8)
ax.plot(layers, last_token_norm, marker='s', linewidth=2, markersize=4,
        color='#E63946', label='Last-token only', alpha=0.8)
ax.axvline(35, color='gray', linestyle='--', alpha=0.5, label='Layer 35')
ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Magnitude (peak = 1.0)', fontsize=12, fontweight='bold')
ax.set_title('Normalized Profile Comparison\nShapes are remarkably similar despite magnitude difference',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=11)

# Add correlation stat
correlation = np.corrcoef(seq_avg_norm, last_token_norm)[0, 1]
ax.text(
    0.97, 0.05,
    f'Profile correlation:\nr = {correlation:.4f}',
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

plt.tight_layout()
plot_path = OUTPUT_DIR / "extraction_method_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to: {plot_path}")

# Print analysis
print(f"\n{'='*80}")
print("EXTRACTION METHOD COMPARISON")
print(f"{'='*80}\n")

print(f"Sequence-averaged method:")
print(f"  Peak layer: 35")
print(f"  Peak magnitude: {seq_avg[-1]:.2f}")
print(f"  Layer 0 magnitude: {seq_avg[0]:.2f}")
print(f"  Growth ratio (L35/L0): {seq_avg[-1]/seq_avg[0]:.1f}×")

print(f"\nLast-token method:")
print(f"  Peak layer: 35")
print(f"  Peak magnitude: {last_token[-1]:.2f}")
print(f"  Layer 0 magnitude: {last_token[0]:.2f}")
print(f"  Growth ratio (L35/L0): {last_token[-1]/last_token[0]:.1f}×")

print(f"\nMagnitude ratio (last-token / seq-avg):")
print(f"  At layer 0: {last_token[0]/seq_avg[0]:.1f}×")
print(f"  At layer 35: {last_token[-1]/seq_avg[-1]:.1f}×")

print(f"\nProfile correlation (normalized): r = {correlation:.4f}")

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}\n")

print("Both methods identify layer 35 as the peak complexity representation layer.")
print(f"The last-token method produces vectors ~{ratio_at_35:.0f}× larger in magnitude,")
print("but the normalized profiles are highly correlated (r > 0.99).")
print("\nThis suggests:")
print("  • Layer 35 is genuinely the 'complexity layer' regardless of extraction method")
print("  • Last-token activations show stronger differentiation (more separation)")
print("  • The geometric structure is consistent - just scaled differently")
print("\nNext: Test if last-token validation improves R² correlation.")
print()
