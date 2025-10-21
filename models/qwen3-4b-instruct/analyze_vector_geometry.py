#!/usr/bin/env python3
"""
Analyze geometric properties of the complexity vector V_c.

Investigates whether V_c is axis-aligned (dominated by single dimension)
or oblique (spread across many dimensions).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the complexity vector
vector_path = Path("output/complexity_vectors.pt")
data = torch.load(vector_path)

# Extract the steering layer vector (usually layer 35)
if isinstance(data, dict):
    # Find the layer with maximum magnitude
    layer_magnitudes = {k: torch.norm(v).item() for k, v in data.items() if isinstance(v, torch.Tensor)}
    max_layer = max(layer_magnitudes, key=layer_magnitudes.get)
    v_c = data[max_layer].float().cpu().numpy().flatten()
    print(f"Using layer: {max_layer}")
    print(f"Layer magnitudes: {layer_magnitudes}")
else:
    v_c = data.cpu().numpy().flatten()

print(f"\n{'='*80}")
print("COMPLEXITY VECTOR GEOMETRY ANALYSIS")
print(f"{'='*80}\n")

# Basic properties
n_dims = len(v_c)
l2_norm = np.linalg.norm(v_c)
l1_norm = np.sum(np.abs(v_c))
l_inf_norm = np.max(np.abs(v_c))

print(f"Dimensionality: {n_dims}")
print(f"L2 norm (||v||₂): {l2_norm:.4f}")
print(f"L1 norm (||v||₁): {l1_norm:.4f}")
print(f"L∞ norm (||v||∞): {l_inf_norm:.4f}")

# Normalize for analysis
v_normalized = v_c / l2_norm
v_squared = v_normalized ** 2

print(f"\n{'='*80}")
print("AXIS ALIGNMENT ANALYSIS")
print(f"{'='*80}\n")

# 1. L∞/L2 ratio (if close to 1, dominated by single component)
alignment_ratio = l_inf_norm / l2_norm
print(f"L∞/L2 ratio: {alignment_ratio:.6f}")
if alignment_ratio > 0.9:
    print("  → Highly axis-aligned (dominated by single dimension)")
elif alignment_ratio > 0.5:
    print("  → Moderately concentrated")
else:
    print("  → Oblique/diffuse (spread across many dimensions)")

# 2. Participation Ratio (effective number of dimensions)
# PR = 1 / Σ(v_i^2)^2 where v is normalized
participation_ratio = 1.0 / np.sum(v_squared ** 2)
print(f"\nParticipation Ratio: {participation_ratio:.2f} / {n_dims}")
print(f"  → Vector effectively spans {participation_ratio:.0f} dimensions")
print(f"  → That's {100*participation_ratio/n_dims:.2f}% of total dimensionality")

if participation_ratio < 10:
    print("  → Highly sparse (axis-aligned)")
elif participation_ratio < 100:
    print("  → Moderately sparse")
elif participation_ratio < n_dims / 2:
    print("  → Moderately diffuse")
else:
    print("  → Highly diffuse (oblique)")

# 3. Top-k concentration
print(f"\n{'='*80}")
print("CONCENTRATION IN TOP COMPONENTS")
print(f"{'='*80}\n")

sorted_squared = np.sort(v_squared)[::-1]  # Descending
cumsum = np.cumsum(sorted_squared)

for k in [1, 5, 10, 50, 100, 500]:
    if k <= n_dims:
        variance_captured = cumsum[k-1]
        print(f"Top {k:4d} components: {100*variance_captured:6.2f}% of ||v||²")

# 4. Sparsity (how many near-zero components?)
print(f"\n{'='*80}")
print("SPARSITY ANALYSIS")
print(f"{'='*80}\n")

thresholds = [0.001, 0.01, 0.1]
for thresh in thresholds:
    n_significant = np.sum(np.abs(v_normalized) > thresh)
    pct_significant = 100 * n_significant / n_dims
    print(f"Components with |v_i| > {thresh}: {n_significant:5d} ({pct_significant:5.2f}%)")

# 5. Statistical measures
print(f"\n{'='*80}")
print("STATISTICAL MEASURES")
print(f"{'='*80}\n")

mean = np.mean(v_normalized)
std = np.std(v_normalized)
kurtosis = np.mean((v_normalized - mean)**4) / (std**4) - 3  # Excess kurtosis
skewness = np.mean((v_normalized - mean)**3) / (std**3)

print(f"Mean: {mean:.6f}")
print(f"Std Dev: {std:.6f}")
print(f"Kurtosis (excess): {kurtosis:.2f}")
if kurtosis > 3:
    print("  → Highly concentrated (leptokurtic)")
elif kurtosis > 0:
    print("  → Moderately concentrated")
else:
    print("  → Uniformly spread (platykurtic)")
print(f"Skewness: {skewness:.2f}")

# 6. Find dominant components
print(f"\n{'='*80}")
print("TOP 10 DOMINANT DIMENSIONS")
print(f"{'='*80}\n")

top_indices = np.argsort(np.abs(v_c))[::-1][:10]
print(f"{'Rank':<6} {'Dimension':<12} {'Value':<12} {'% of L2':<10}")
print("-"*50)
for rank, idx in enumerate(top_indices, 1):
    value = v_c[idx]
    pct = 100 * abs(value) / l2_norm
    print(f"{rank:<6} {idx:<12} {value:>11.6f} {pct:>9.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sorted component magnitudes (log scale)
ax = axes[0, 0]
sorted_abs = np.sort(np.abs(v_normalized))[::-1]
ax.plot(sorted_abs, linewidth=1.5, color='#2E86AB')
ax.set_xlabel('Component Rank', fontweight='bold')
ax.set_ylabel('|Component Value|', fontweight='bold')
ax.set_title('Component Magnitudes (Sorted)', fontweight='bold', fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')
ax.legend()

# 2. Cumulative variance explained
ax = axes[0, 1]
ax.plot(cumsum, linewidth=2, color='#F18F01')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% of variance')
ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90% of variance')
ax.set_xlabel('Number of Top Components', fontweight='bold')
ax.set_ylabel('Cumulative % of ||v||²', fontweight='bold')
ax.set_title('Cumulative Variance Captured', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()

# 3. Histogram of component values
ax = axes[1, 0]
ax.hist(v_normalized, bins=50, color='#06A77D', alpha=0.7, edgecolor='black')
ax.set_xlabel('Component Value', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Component Values', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 4. Top components bar chart
ax = axes[1, 1]
top_20_indices = np.argsort(np.abs(v_c))[::-1][:20]
top_20_values = v_c[top_20_indices]
colors = ['#E63946' if x < 0 else '#06A77D' for x in top_20_values]
ax.bar(range(20), top_20_values, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Top 20 Dimensions (by magnitude)', fontweight='bold')
ax.set_ylabel('Component Value', fontweight='bold')
ax.set_title('Top 20 Components', fontweight='bold', fontsize=12)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = Path("output/vector_geometry_analysis.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

if participation_ratio < 50:
    concentration = "SPARSE/CONCENTRATED"
    interpretation = f"V_c is dominated by ~{int(participation_ratio)} key dimensions"
elif participation_ratio < n_dims / 4:
    concentration = "MODERATELY DIFFUSE"
    interpretation = f"V_c spans ~{int(participation_ratio)} dimensions meaningfully"
else:
    concentration = "HIGHLY DIFFUSE"
    interpretation = f"V_c is spread across {int(participation_ratio)}+ dimensions"

print(f"Vector character: {concentration}")
print(f"Interpretation: {interpretation}")
print(f"\nIn your notation:")
if participation_ratio < 10:
    print("  → V_c ≈ 'almost parallel to dimension {top dimension}'")
elif participation_ratio < 100:
    print(f"  → V_c ≈ 'concentrated in a ~{int(participation_ratio)}-dimensional subspace'")
else:
    print("  → V_c ≈ 'highly oblique, pointed off in a complex direction'")
print()
