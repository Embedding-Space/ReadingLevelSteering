#!/usr/bin/env python3
"""
Piecewise analysis of complexity vector coherence across layer ranges.

Questions:
- Early layers (0-5): Are vectors all over the place (pincushion)?
- Middle layers (6-29): Do vectors converge to co-linearity?
- Late layers (30-35): What's happening at the end?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
VECTORS_PATH = OUTPUT_DIR / "complexity_vectors.pt"

# Layer ranges to analyze
RANGES = {
    'Early (0-5)': range(0, 6),
    'Middle (6-29)': range(6, 30),
    'Late (30-35)': range(30, 36),
}

print(f"\n{'='*80}")
print("PIECEWISE CROSS-LAYER COHERENCE ANALYSIS")
print(f"{'='*80}\n")

# Load all complexity vectors
vectors_dict = torch.load(VECTORS_PATH)
num_layers = len(vectors_dict)

# Convert to numpy array: (num_layers, hidden_dim)
all_vectors = np.array([vectors_dict[i].float().cpu().numpy().flatten()
                        for i in range(num_layers)])

print(f"Loaded {num_layers} complexity vectors")
print(f"Dimensionality: {all_vectors.shape[1]}\n")

# Analyze each range
results = {}

for range_name, layer_range in RANGES.items():
    print(f"{'='*80}")
    print(f"{range_name}")
    print(f"{'='*80}\n")

    # Extract vectors for this range
    layers_in_range = list(layer_range)
    vectors = all_vectors[layers_in_range]
    n = len(vectors)

    print(f"Layers: {layers_in_range}")
    print(f"Number of vectors: {n}\n")

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    # Cosine similarity matrix
    cosine_sim_matrix = normalized_vectors @ normalized_vectors.T

    # Pairwise similarities (upper triangle, excluding diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sims = cosine_sim_matrix[upper_tri_indices]

    if len(pairwise_sims) > 0:
        mean_sim = pairwise_sims.mean()
        std_sim = pairwise_sims.std()
        min_sim = pairwise_sims.min()
        max_sim = pairwise_sims.max()
        median_sim = np.median(pairwise_sims)

        print(f"Pairwise cosine similarities:")
        print(f"  Mean:   {mean_sim:.4f}")
        print(f"  Std:    {std_sim:.4f}")
        print(f"  Min:    {min_sim:.4f}")
        print(f"  Max:    {max_sim:.4f}")
        print(f"  Median: {median_sim:.4f}")

        # Centroid analysis
        centroid = normalized_vectors.mean(axis=0)
        centroid_norm_before = np.linalg.norm(centroid)
        centroid_normalized = centroid / centroid_norm_before

        similarities_to_centroid = normalized_vectors @ centroid_normalized
        angular_deviations = np.arccos(np.clip(similarities_to_centroid, -1, 1)) * 180 / np.pi

        print(f"\nCentroid analysis:")
        print(f"  Centroid L2 norm (before normalization): {centroid_norm_before:.4f}")
        print(f"  Mean similarity to centroid: {similarities_to_centroid.mean():.4f}")
        print(f"  Mean angular deviation: {angular_deviations.mean():.2f}°")
        print(f"  Max angular deviation: {angular_deviations.max():.2f}°")

        # Interpretation
        print(f"\nInterpretation:")
        if mean_sim > 0.95:
            print(f"  ✓✓✓ EXTREMELY HIGH coherence - nearly co-linear!")
            print(f"      Vectors are pointing in almost exactly the same direction")
        elif mean_sim > 0.85:
            print(f"  ✓✓ VERY HIGH coherence - tightly clustered")
            print(f"     Vectors form a tight cone in activation space")
        elif mean_sim > 0.7:
            print(f"  ✓ HIGH coherence - pointing in similar directions")
            print(f"    Clear shared complexity direction emerging")
        elif mean_sim > 0.5:
            print(f"  ~ MODERATE coherence - related but with variation")
            print(f"    Complexity representation is developing")
        else:
            print(f"  ✗ LOW coherence - scattered like a pincushion!")
            print(f"    Vectors point in substantially different directions")

        # Store results
        results[range_name] = {
            'layers': layers_in_range,
            'n': n,
            'mean_pairwise_sim': mean_sim,
            'std_pairwise_sim': std_sim,
            'min_pairwise_sim': min_sim,
            'max_pairwise_sim': max_sim,
            'median_pairwise_sim': median_sim,
            'centroid_norm': centroid_norm_before,
            'mean_angular_deviation': angular_deviations.mean(),
            'max_angular_deviation': angular_deviations.max(),
        }
    else:
        print("  (Only one layer - no pairwise comparisons possible)")
        results[range_name] = {'n': n, 'layers': layers_in_range}

    print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Mean pairwise similarity by range
ax = axes[0, 0]
range_names = list(results.keys())
mean_sims = [results[r]['mean_pairwise_sim'] for r in range_names if 'mean_pairwise_sim' in results[r]]
colors = ['#E63946', '#F18F01', '#06A77D']

bars = ax.bar(range(len(mean_sims)), mean_sims, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Near co-linear (>0.95)')
ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='High coherence (>0.7)')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate (>0.5)')
ax.set_xticks(range(len(range_names)))
ax.set_xticklabels(range_names, fontsize=11)
ax.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=12, fontweight='bold')
ax.set_title('Coherence by Layer Range\nHigher = more co-linear',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim([0, 1])
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, mean_sims)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Angular deviation distributions
ax = axes[0, 1]
for i, (range_name, color) in enumerate(zip(range_names, colors)):
    if 'mean_angular_deviation' in results[range_name]:
        layers = results[range_name]['layers']
        vectors = all_vectors[layers]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms

        centroid = normalized_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        centroid_normalized = centroid / centroid_norm

        similarities = normalized_vectors @ centroid_normalized
        deviations = np.arccos(np.clip(similarities, -1, 1)) * 180 / np.pi

        ax.hist(deviations, bins=15, alpha=0.6, color=color, label=range_name, edgecolor='black')

ax.set_xlabel('Angular Deviation from Range Centroid (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('How Scattered Are Vectors Within Each Range?',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 3: Standard deviation comparison
ax = axes[1, 0]
std_sims = [results[r]['std_pairwise_sim'] for r in range_names if 'std_pairwise_sim' in results[r]]
bars = ax.bar(range(len(std_sims)), std_sims, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(range_names)))
ax.set_xticklabels(range_names, fontsize=11)
ax.set_ylabel('Std Dev of Pairwise Similarities', fontsize=12, fontweight='bold')
ax.set_title('Variability Within Each Range\nLower = more consistent',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, std_sims)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Centroid norm comparison
ax = axes[1, 1]
centroid_norms = [results[r]['centroid_norm'] for r in range_names if 'centroid_norm' in results[r]]
bars = ax.bar(range(len(centroid_norms)), centroid_norms, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(range_names)))
ax.set_xticklabels(range_names, fontsize=11)
ax.set_ylabel('Centroid L2 Norm (before normalization)', fontsize=12, fontweight='bold')
ax.set_title('How "Tight" is the Cluster?\nHigher = vectors pointing in similar direction',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

for i, (bar, val) in enumerate(zip(bars, centroid_norms)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_path = OUTPUT_DIR / "cross_layer_coherence_piecewise.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {plot_path}\n")

# Summary comparison
print(f"{'='*80}")
print("SUMMARY COMPARISON")
print(f"{'='*80}\n")

print(f"{'Range':<20} {'Mean Sim':<12} {'Std':<12} {'Centroid':<12} {'Mean Δ°':<12}")
print(f"{'-'*80}")
for range_name in range_names:
    if 'mean_pairwise_sim' in results[range_name]:
        r = results[range_name]
        print(f"{range_name:<20} {r['mean_pairwise_sim']:>10.4f}  {r['std_pairwise_sim']:>10.4f}  "
              f"{r['centroid_norm']:>10.4f}  {r['mean_angular_deviation']:>10.2f}°")

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}\n")

early_sim = results['Early (0-5)']['mean_pairwise_sim']
middle_sim = results['Middle (6-29)']['mean_pairwise_sim']
late_sim = results['Late (30-35)']['mean_pairwise_sim']

print(f"Early layers (0-5):   Mean similarity = {early_sim:.3f}")
if early_sim < 0.5:
    print(f"  → PINCUSHION! Vectors point in many different directions")
    print(f"  → Complexity representation hasn't emerged yet")
else:
    print(f"  → Some structure emerging but still scattered")

print(f"\nMiddle layers (6-29): Mean similarity = {middle_sim:.3f}")
if middle_sim > 0.85:
    print(f"  → HIGHLY CO-LINEAR! Clear convergence on shared direction")
    print(f"  → This is where the model represents complexity coherently")
elif middle_sim > 0.7:
    print(f"  → Strong coherence, vectors mostly aligned")
else:
    print(f"  → Still developing, moderate alignment")

print(f"\nLate layers (30-35):  Mean similarity = {late_sim:.3f}")
if late_sim > middle_sim + 0.05:
    print(f"  → EVEN TIGHTER! Final layers refine the representation")
elif late_sim > middle_sim - 0.05:
    print(f"  → MAINTAINED! Coherence stays strong through final layers")
else:
    print(f"  → Divergence at the end (task-specific adjustments?)")

print()
