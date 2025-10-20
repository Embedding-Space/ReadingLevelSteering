"""
Prompted FK Vector Extraction

Extracts the "prompted reading level" vector by capturing activations when the
model is explicitly asked to write at different grade levels, then compares this
to our extracted steering vector.

Research question: When models try to follow grade-level instructions, do they
activate the same geometric direction as our steering vector? Or something else?

Experimental design:
- Generate responses at low grade (5th) and high grade (20th) via prompting
- Capture activations at all layers during generation
- Extract PFK vector = mean(high_grade) - mean(low_grade)
- Analyze vector properties (magnitude, concentration/sparsity)
- Compare to steering vector (cosine similarity, magnitude ratio)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from typing import Dict, List, Tuple
from scipy.stats import entropy
import matplotlib.pyplot as plt


# Configuration
MODEL_NAME = "ibm-granite/granite-4.0-micro"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("./output")
STEERING_VECTORS_PATH = OUTPUT_DIR / "complexity_vectors.pt"

# Test configuration - use multiple prompts for robustness
TEST_PROMPTS = [
    "Can you explain quantum mechanics please?",
    "Tell me about the theory of relativity",
    "Explain how photosynthesis works",
    "Describe the water cycle",
    "What is climate change?",
]

LOW_GRADE = 5
HIGH_GRADE = 20
MAX_NEW_TOKENS = 200
SEED = 42


def setup_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")


def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print(f"\n{'='*80}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map=DEVICE,
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"✓ Model loaded successfully")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Hidden dimension: {model.config.hidden_size}")

    return model, tokenizer, num_layers


def capture_activations_during_generation(
    model,
    tokenizer,
    prompt: str,
    grade_level: int,
    num_layers: int
) -> Dict[int, torch.Tensor]:
    """
    Generate text with grade level prompt and capture activations.

    Returns activations averaged across all generated tokens.
    """
    # Storage for activations
    layer_activations = {i: [] for i in range(num_layers)}

    def create_hook(layer_idx):
        def hook(module, input, output):
            # output shape: (batch_size, seq_len, hidden_dim)
            # We'll collect activations from all generated tokens
            layer_activations[layer_idx].append(output[0].detach().cpu())
        return hook

    # Register hooks
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(create_hook(i))
        hooks.append(hook)

    # Format prompt
    formatted_prompt = f"Please explain the following at a {grade_level}th grade reading level: {prompt}"
    messages = [{"role": "user", "content": formatted_prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy decoding for determinism
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Average activations across all tokens for each layer
    averaged_activations = {}
    for layer_idx in range(num_layers):
        # Concatenate all activations from this layer across generation steps
        all_acts = torch.cat(layer_activations[layer_idx], dim=0)  # (total_tokens, hidden_dim)
        # Average across tokens
        averaged_activations[layer_idx] = all_acts.mean(dim=0)  # (hidden_dim,)

    return averaged_activations


def extract_prompted_vector(
    model,
    tokenizer,
    num_layers: int
) -> Dict[int, torch.Tensor]:
    """
    Extract PFK vector by comparing low-grade vs high-grade prompted activations.
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING PROMPTED FK VECTOR")
    print(f"{'='*80}\n")
    print(f"Low grade: {LOW_GRADE}th")
    print(f"High grade: {HIGH_GRADE}th")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print()

    low_grade_activations = {i: [] for i in range(num_layers)}
    high_grade_activations = {i: [] for i in range(num_layers)}

    # Collect activations from all test prompts
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{i}/{len(TEST_PROMPTS)}] Processing: '{prompt[:50]}...'")

        # Low grade
        print(f"  - Grade {LOW_GRADE}...", end=" ", flush=True)
        low_acts = capture_activations_during_generation(
            model, tokenizer, prompt, LOW_GRADE, num_layers
        )
        for layer_idx in range(num_layers):
            low_grade_activations[layer_idx].append(low_acts[layer_idx])
        print("✓")

        # High grade
        print(f"  - Grade {HIGH_GRADE}...", end=" ", flush=True)
        high_acts = capture_activations_during_generation(
            model, tokenizer, prompt, HIGH_GRADE, num_layers
        )
        for layer_idx in range(num_layers):
            high_grade_activations[layer_idx].append(high_acts[layer_idx])
        print("✓")

    # Compute mean across prompts and subtract
    print(f"\nComputing prompted FK vectors...")
    prompted_vectors = {}

    for layer_idx in range(num_layers):
        # Stack and average
        low_mean = torch.stack(low_grade_activations[layer_idx]).mean(dim=0)
        high_mean = torch.stack(high_grade_activations[layer_idx]).mean(dim=0)

        # Subtract: high - low (positive = more complex)
        prompted_vectors[layer_idx] = high_mean - low_mean

    print(f"✓ Extracted {len(prompted_vectors)} layer vectors")

    return prompted_vectors


def analyze_vector_concentration(v: torch.Tensor) -> Dict[str, float]:
    """
    Analyze how concentrated/sparse a vector is.

    Returns metrics describing the distribution of vector components.
    """
    v_np = v.float().cpu().numpy() if torch.is_tensor(v) else v
    v_abs = np.abs(v_np)
    v_sq = v_np ** 2

    # Magnitude (L2 norm)
    magnitude = np.linalg.norm(v_np)

    # Participation ratio (normalized)
    # PR near 0 = concentrated, near 1 = diffuse
    PR = (np.sum(v_sq) ** 2) / np.sum(v_sq ** 2)
    PR_normalized = PR / len(v_np)

    # Effective dimensionality via Shannon entropy
    v_prob = v_abs / (np.sum(v_abs) + 1e-10)
    eff_dim = np.exp(entropy(v_prob + 1e-10))
    eff_dim_normalized = eff_dim / len(v_np)

    # Top-k concentration (what % of magnitude in top 10% of components?)
    sorted_v = np.sort(v_abs)[::-1]
    k = max(1, int(0.1 * len(v_np)))  # Top 10%
    top_k_magnitude = np.sum(sorted_v[:k])
    top_k_ratio = top_k_magnitude / (np.sum(v_abs) + 1e-10)

    return {
        'magnitude': float(magnitude),
        'participation_ratio': float(PR_normalized),
        'effective_dim': float(eff_dim_normalized),
        'top_10_pct_concentration': float(top_k_ratio),
    }


def compare_vectors(
    prompted_vector: torch.Tensor,
    steering_vector: torch.Tensor,
    layer_idx: int
) -> Dict[str, float]:
    """
    Compare prompted FK vector to steering vector.
    """
    v1_np = prompted_vector.float().cpu().numpy() if torch.is_tensor(prompted_vector) else prompted_vector
    v2_np = steering_vector.float().cpu().numpy() if torch.is_tensor(steering_vector) else steering_vector

    # Cosine similarity
    cos_sim = np.dot(v1_np, v2_np) / (np.linalg.norm(v1_np) * np.linalg.norm(v2_np) + 1e-10)

    # Magnitude ratio
    mag_ratio = np.linalg.norm(v1_np) / (np.linalg.norm(v2_np) + 1e-10)

    # Analyze concentration for both
    c1 = analyze_vector_concentration(v1_np)
    c2 = analyze_vector_concentration(v2_np)

    return {
        'layer': layer_idx,
        'cosine_similarity': float(cos_sim),
        'magnitude_ratio': float(mag_ratio),
        'prompted_magnitude': c1['magnitude'],
        'steering_magnitude': c2['magnitude'],
        'prompted_participation_ratio': c1['participation_ratio'],
        'steering_participation_ratio': c2['participation_ratio'],
        'prompted_effective_dim': c1['effective_dim'],
        'steering_effective_dim': c2['effective_dim'],
        'prompted_top10_concentration': c1['top_10_pct_concentration'],
        'steering_top10_concentration': c2['top_10_pct_concentration'],
    }


def main():
    """Run the full prompted FK vector extraction and comparison."""
    torch.manual_seed(SEED)
    setup_output_dir()

    # Load model
    model, tokenizer, num_layers = load_model_and_tokenizer()

    # Extract prompted FK vector
    prompted_vectors = extract_prompted_vector(model, tokenizer, num_layers)

    # Save prompted vectors
    prompted_vectors_path = OUTPUT_DIR / "prompted_fk_vectors.pt"
    torch.save(prompted_vectors, prompted_vectors_path)
    print(f"\n✓ Saved prompted FK vectors to: {prompted_vectors_path}")

    # Load steering vectors for comparison
    print(f"\n{'='*80}")
    print(f"COMPARING TO STEERING VECTORS")
    print(f"{'='*80}\n")

    if not STEERING_VECTORS_PATH.exists():
        print(f"⚠ Steering vectors not found at {STEERING_VECTORS_PATH}")
        print(f"  Run extract_complexity_vectors.py first!")
        return

    steering_vectors = torch.load(STEERING_VECTORS_PATH)
    print(f"✓ Loaded steering vectors from: {STEERING_VECTORS_PATH}")

    # Compare at each layer
    results = []
    for layer_idx in range(num_layers):
        comparison = compare_vectors(
            prompted_vectors[layer_idx],
            steering_vectors[layer_idx],
            layer_idx
        )
        results.append(comparison)

    # Find most interesting layer (highest cosine similarity)
    best_layer = max(results, key=lambda x: x['cosine_similarity'])

    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(f"Best layer (highest cosine similarity): {best_layer['layer']}")
    print(f"  Cosine similarity: {best_layer['cosine_similarity']:.4f}")
    print(f"  Magnitude ratio (prompted/steering): {best_layer['magnitude_ratio']:.4f}")
    print(f"\n  Prompted FK vector:")
    print(f"    - Magnitude: {best_layer['prompted_magnitude']:.2f}")
    print(f"    - Participation ratio: {best_layer['prompted_participation_ratio']:.4f}")
    print(f"    - Effective dimensionality: {best_layer['prompted_effective_dim']:.4f}")
    print(f"    - Top-10% concentration: {best_layer['prompted_top10_concentration']:.2%}")
    print(f"\n  Steering vector:")
    print(f"    - Magnitude: {best_layer['steering_magnitude']:.2f}")
    print(f"    - Participation ratio: {best_layer['steering_participation_ratio']:.4f}")
    print(f"    - Effective dimensionality: {best_layer['steering_effective_dim']:.4f}")
    print(f"    - Top-10% concentration: {best_layer['steering_top10_concentration']:.2%}")

    # Save comparison results
    import pandas as pd
    df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "prompted_vs_steering_comparison.csv"
    df.to_csv(results_path, index=False)
    print(f"\n✓ Saved comparison results to: {results_path}")

    # Visualize comparison across layers
    visualize_comparison(results, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")


def visualize_comparison(results: List[Dict], output_dir: Path):
    """Create visualization comparing prompted vs steering vectors across layers."""
    layers = [r['layer'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Cosine similarity across layers
    ax = axes[0, 0]
    ax.plot(layers, [r['cosine_similarity'] for r in results],
            'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Prompted vs Steering Vector Similarity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Magnitude comparison
    ax = axes[0, 1]
    ax.plot(layers, [r['prompted_magnitude'] for r in results],
            'o-', color='#F18F01', linewidth=2, markersize=6, label='Prompted FK')
    ax.plot(layers, [r['steering_magnitude'] for r in results],
            's-', color='#A23B72', linewidth=2, markersize=6, label='Steering')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vector Magnitude (L2 norm)', fontsize=12, fontweight='bold')
    ax.set_title('Vector Magnitude Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Concentration (participation ratio)
    ax = axes[1, 0]
    ax.plot(layers, [r['prompted_participation_ratio'] for r in results],
            'o-', color='#F18F01', linewidth=2, markersize=6, label='Prompted FK')
    ax.plot(layers, [r['steering_participation_ratio'] for r in results],
            's-', color='#A23B72', linewidth=2, markersize=6, label='Steering')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Participation Ratio (normalized)', fontsize=12, fontweight='bold')
    ax.set_title('Vector Concentration (lower = more concentrated)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Effective dimensionality
    ax = axes[1, 1]
    ax.plot(layers, [r['prompted_effective_dim'] for r in results],
            'o-', color='#F18F01', linewidth=2, markersize=6, label='Prompted FK')
    ax.plot(layers, [r['steering_effective_dim'] for r in results],
            's-', color='#A23B72', linewidth=2, markersize=6, label='Steering')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effective Dimensionality (normalized)', fontsize=12, fontweight='bold')
    ax.set_title('Vector Sparsity (lower = more sparse)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "prompted_vs_steering_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")


if __name__ == "__main__":
    main()
