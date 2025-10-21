#!/usr/bin/env python3
"""
Analyze how similar complexity vectors are across all 36 layers.

Questions:
- Are all layers pointing in roughly the same direction?
- Is there a "centroid" complexity direction?
- Do certain layers cluster together?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
VECTORS_PATH = OUTPUT_DIR / "complexity_vectors.pt"

print(f"\n{'='*80}")
print("CROSS-LAYER COMPLEXITY VECTOR COHERENCE ANALYSIS")
print(f"{'='*80}\n")

# Load all complexity vectors
vectors_dict = torch.load(VECTORS_PATH)
num_layers = len(vectors_dict)

# Convert to numpy array: (num_layers, hidden_dim)
vectors = np.array([vectors_dict[i].float().cpu().numpy().flatten()
                    for i in range(num_layers)])

print(f"Loaded {num_layers} complexity vectors")
print(f"Dimensionality: {vectors.shape[1]}\n")

# Compute pairwise cosine similarities
print(f"{'='*80}")
print("PAIRWISE COSINE SIMILARITY")
print(f"{'='*80}\n")

# Normalize vectors
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized_vectors = vectors / norms

# Cosine similarity matrix: (num_layers, num_layers)
cosine_sim_matrix = normalized_vectors @ normalized_vectors.T

# Compute statistics
# Upper triangle only (exclude diagonal)
upper_tri_indices = np.triu_indices(num_layers, k=1)
pairwise_sims = cosine_sim_matrix[upper_tri_indices]

print(f"Pairwise cosine similarities (excluding self-similarity):")
print(f"  Mean: {pairwise_sims.mean():.4f}")
print(f"  Std:  {pairwise_sims.std():.4f}")
print(f"  Min:  {pairwise_sims.min():.4f} (layers {np.unravel_index(cosine_sim_matrix.argmin(), cosine_sim_matrix.shape)})")
print(f"  Max:  {pairwise_sims.max():.4f}")
print(f"  Median: {np.median(pairwise_sims):.4f}")

# Find most similar and most dissimilar pairs
most_similar_idx = np.unravel_index(
    np.ma.masked_array(cosine_sim_matrix, mask=np.eye(num_layers, dtype=bool)).argmax(),
    cosine_sim_matrix.shape
)
most_dissimilar_idx = np.unravel_index(cosine_sim_matrix.argmin(), cosine_sim_matrix.shape)

print(f"\nMost similar pair: Layers {most_similar_idx[0]} and {most_similar_idx[1]}")
print(f"  Cosine similarity: {cosine_sim_matrix[most_similar_idx]:.4f}")
print(f"Most dissimilar pair: Layers {most_dissimilar_idx[0]} and {most_dissimilar_idx[1]}")
print(f"  Cosine similarity: {cosine_sim_matrix[most_dissimilar_idx]:.4f}")

# Centroid analysis
print(f"\n{'='*80}")
print("CENTROID ANALYSIS")
print(f"{'='*80}\n")

# Compute centroid (mean vector across all layers)
centroid = normalized_vectors.mean(axis=0)
centroid_norm = np.linalg.norm(centroid)
centroid_normalized = centroid / centroid_norm

print(f"Centroid L2 norm (before normalization): {centroid_norm:.4f}")

# Compute cosine similarity of each layer to centroid
similarities_to_centroid = normalized_vectors @ centroid_normalized

print(f"\nCosine similarity to centroid:")
print(f"  Mean: {similarities_to_centroid.mean():.4f}")
print(f"  Std:  {similarities_to_centroid.std():.4f}")
print(f"  Min:  {similarities_to_centroid.min():.4f} (layer {similarities_to_centroid.argmin()})")
print(f"  Max:  {similarities_to_centroid.max():.4f} (layer {similarities_to_centroid.argmax()})")

# Angular deviation from centroid
angular_deviations = np.arccos(np.clip(similarities_to_centroid, -1, 1)) * 180 / np.pi

print(f"\nAngular deviation from centroid:")
print(f"  Mean: {angular_deviations.mean():.2f}°")
print(f"  Std:  {angular_deviations.std():.2f}°")
print(f"  Max:  {angular_deviations.max():.2f}° (layer {angular_deviations.argmax()})")

# Interpretation
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}\n")

mean_sim = pairwise_sims.mean()
if mean_sim > 0.9:
    print("✓ VERY HIGH coherence: All layers point in nearly the same direction")
    print("  → Complexity is represented consistently across all layers")
elif mean_sim > 0.7:
    print("✓ HIGH coherence: Most layers point in similar directions")
    print("  → Clear emergence of a coherent complexity dimension")
elif mean_sim > 0.5:
    print("~ MODERATE coherence: Layers are related but with variation")
    print("  → Complexity emerges gradually across layers")
else:
    print("✗ LOW coherence: Layers point in substantially different directions")
    print("  → No unified complexity representation")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Cosine similarity heatmap
ax = axes[0, 0]
im = ax.imshow(cosine_sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_title(
    f'Pairwise Cosine Similarity Matrix\nMean similarity: {mean_sim:.3f}',
    fontsize=13, fontweight='bold', pad=15
)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Cosine Similarity', fontsize=11)
ax.set_xticks(np.arange(0, num_layers, 5))
ax.set_yticks(np.arange(0, num_layers, 5))

# Plot 2: Histogram of pairwise similarities
ax = axes[0, 1]
ax.hist(pairwise_sims, bins=30, color='#06A77D', alpha=0.7, edgecolor='black')
ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_sim:.3f}')
ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Pairwise Cosine Similarities',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 3: Similarity to centroid by layer
ax = axes[1, 0]
ax.plot(range(num_layers), similarities_to_centroid, marker='o',
        linewidth=2, markersize=6, color='#E63946')
ax.axhline(similarities_to_centroid.mean(), color='gray', linestyle='--',
           label=f'Mean: {similarities_to_centroid.mean():.3f}')
ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Cosine Similarity to Centroid', fontsize=12, fontweight='bold')
ax.set_title('How Similar is Each Layer to the Average?',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim([0, 1])

# Plot 4: Hierarchical clustering dendrogram
ax = axes[1, 1]
# Convert similarity to distance for clustering
distance_matrix = 1 - cosine_sim_matrix
condensed_distance = squareform(distance_matrix, checks=False)
linkage_matrix = linkage(condensed_distance, method='average')

dendrogram(
    linkage_matrix,
    ax=ax,
    labels=[str(i) for i in range(num_layers)],
    leaf_font_size=8,
    color_threshold=0.3
)
ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance (1 - cosine similarity)', fontsize=12, fontweight='bold')
ax.set_title('Hierarchical Clustering of Layers\n(Which layers are most similar?)',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plot_path = OUTPUT_DIR / "cross_layer_coherence_analysis.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {plot_path}\n")

# Save numerical results
results = {
    'mean_pairwise_similarity': float(mean_sim),
    'std_pairwise_similarity': float(pairwise_sims.std()),
    'min_pairwise_similarity': float(pairwise_sims.min()),
    'max_pairwise_similarity': float(pairwise_sims.max()),
    'median_pairwise_similarity': float(np.median(pairwise_sims)),
    'mean_similarity_to_centroid': float(similarities_to_centroid.mean()),
    'mean_angular_deviation_degrees': float(angular_deviations.mean()),
    'centroid_norm_before_normalization': float(centroid_norm),
}

import json
results_path = OUTPUT_DIR / "cross_layer_coherence_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Numerical results saved to: {results_path}\n")

print(f"{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")
