#!/usr/bin/env python3
"""
Compare cosine similarity between sequence-averaged and last-token V_c vectors.

Are they pointing in the same direction despite different extraction methods?
"""

import torch
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

SEQ_AVG_PATH = OUTPUT_DIR / "complexity_vectors.pt"
LAST_TOKEN_PATH = OUTPUT_DIR / "complexity_vectors_last_token.pt"

LAYER = 35  # Peak layer for both methods

print(f"\n{'='*80}")
print("COMPARING COMPLEXITY VECTORS")
print(f"{'='*80}\n")

# Load vectors
print("Loading vectors...")
seq_avg_data = torch.load(SEQ_AVG_PATH)
last_token_data = torch.load(LAST_TOKEN_PATH)

v_seq = seq_avg_data[LAYER].float().cpu().numpy().flatten()
v_last = last_token_data[LAYER].float().cpu().numpy().flatten()

print(f"✓ Loaded both vectors from layer {LAYER}")
print(f"  Dimensionality: {len(v_seq)}")

# Compute L2 norms
norm_seq = np.linalg.norm(v_seq)
norm_last = np.linalg.norm(v_last)

print(f"\nVector magnitudes:")
print(f"  Sequence-averaged: {norm_seq:.4f}")
print(f"  Last-token: {norm_last:.4f}")
print(f"  Ratio: {norm_last / norm_seq:.2f}×")

# Compute cosine similarity
cosine_sim = np.dot(v_seq, v_last) / (norm_seq * norm_last)

print(f"\n{'='*80}")
print("COSINE SIMILARITY")
print(f"{'='*80}\n")

print(f"cos(θ) = {cosine_sim:.6f}")
print(f"Angle: {np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi:.2f}°")

# Interpretation
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}\n")

if cosine_sim > 0.9:
    print("✓ VERY HIGH alignment: Vectors point in essentially the same direction")
    print("  → Both methods capture the same underlying complexity dimension")
elif cosine_sim > 0.7:
    print("✓ HIGH alignment: Vectors point in similar directions")
    print("  → Same general concept, but with some differences")
elif cosine_sim > 0.5:
    print("~ MODERATE alignment: Vectors are somewhat related")
    print("  → Capturing related but distinct aspects")
elif cosine_sim > 0:
    print("✗ WEAK alignment: Vectors are only slightly related")
    print("  → Different representations of complexity")
else:
    print("✗ NEGATIVE alignment: Vectors point in opposite directions!")
    print("  → Methods are capturing opposite phenomena")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

print(f"Despite the last-token method showing poor validation (R²=0.12),")
print(f"the two extraction methods produce vectors with cosine similarity = {cosine_sim:.4f}")

if cosine_sim > 0.7:
    print(f"\nThis suggests they ARE capturing the same underlying direction,")
    print(f"but the last-token representation is too noisy/specific for validation.")
    print(f"Steering might still work if the averaged effect across generation aligns.")
else:
    print(f"\nThis suggests they're capturing DIFFERENT aspects of complexity.")
    print(f"The last-token vector may not steer in the same way.")

print()
