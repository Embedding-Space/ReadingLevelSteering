"""
Complexity Vector Extraction from Wikipedia Pairs - LAST TOKEN METHOD

Extracts reading-level steering vectors using ONLY the last token's activation
from each text, rather than sequence averaging.

This tests the hypothesis that complexity representations may peak at different
layers depending on extraction method:
- Sequence averaging: Captures overall "vibe" (peaks at layer 35)
- Last token: Captures next-token prediction focus (may peak earlier)

Experimental design:
- Load 20+ matched Wikipedia pairs (Simple + Regular on same topics)
- Model: Qwen3-4B-Instruct-2507 (36 layers, bf16)
- Capture LAST TOKEN activation at each layer for both Simple and Regular versions
- Compute complexity_vector = mean(regular_last_tokens) - mean(simple_last_tokens)
- Compare peak layer to sequence-averaged method
- Save vectors for steering experiments
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
from tqdm import tqdm


# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path("../../data/wikipedia_pairs/wikipedia_pairs.json")
OUTPUT_DIR = Path("./output")
SEED = 42

# Use first 20 pairs for consistency with original experiment
NUM_PAIRS_TO_USE = 20


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
    print(f"  - Model dtype: {model.dtype}")

    return model, tokenizer, num_layers


def load_wikipedia_pairs() -> List[Dict]:
    """Load the Wikipedia article pairs."""
    print(f"\n{'='*80}")
    print(f"Loading Wikipedia pairs")
    print(f"{'='*80}\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Wikipedia pairs not found at {DATA_PATH}. "
            "Run prepare_wikipedia_pairs.py first!"
        )

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = data['pairs'][:NUM_PAIRS_TO_USE]  # Use first N pairs

    print(f"✓ Loaded {len(pairs)} article pairs")
    print(f"\nSample topics:")
    for i, pair in enumerate(pairs[:5], 1):
        print(f"  {i}. {pair['topic']} (simple: {pair['simple_grade']:.1f}, regular: {pair['regular_grade']:.1f})")
    if len(pairs) > 5:
        print(f"  ... and {len(pairs) - 5} more")

    return pairs


def capture_activations_last_token(model, tokenizer, texts: List[str], label: str) -> Dict[int, torch.Tensor]:
    """
    Capture LAST TOKEN activations from all layers for a list of texts.

    Returns dict mapping layer_idx -> tensor of shape [num_samples, hidden_dim]
    (NO sequence averaging - just the final token's activation)
    """
    print(f"\n{'='*80}")
    print(f"Capturing LAST TOKEN activations for {len(texts)} {label} samples")
    print(f"{'='*80}\n")

    num_layers = len(model.model.layers)
    layer_activations = {i: [] for i in range(num_layers)}

    with torch.no_grad():
        for idx, text in enumerate(tqdm(texts, desc=f"Processing {label} samples")):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Register hooks to capture layer outputs
            activations = {}

            def make_hook(layer_idx):
                def hook(module, input, output):
                    activations[layer_idx] = output[0] if isinstance(output, tuple) else output
                return hook

            # Register hooks
            hooks = []
            for layer_idx, layer in enumerate(model.model.layers):
                hook = layer.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)

            # Forward pass
            _ = model(**inputs)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Extract LAST TOKEN activation for each layer (no averaging!)
            for layer_idx in range(num_layers):
                last_token_activation = activations[layer_idx][:, -1, :].squeeze(0)
                layer_activations[layer_idx].append(last_token_activation.cpu())

    # Stack activations for each layer
    stacked_activations = {}
    for layer_idx in range(num_layers):
        stacked_activations[layer_idx] = torch.stack(layer_activations[layer_idx])

    print(f"✓ Captured last-token activations from {num_layers} layers")
    print(f"  - Activation shape per layer: {stacked_activations[0].shape}")

    return stacked_activations


def compute_complexity_vectors(
    simple_activations: Dict[int, torch.Tensor],
    regular_activations: Dict[int, torch.Tensor]
) -> tuple[Dict[int, torch.Tensor], Dict[int, float]]:
    """
    Compute complexity vectors and their magnitudes.

    complexity_vector = mean(regular_last_tokens) - mean(simple_last_tokens)

    Returns:
        - complexity_vectors: dict mapping layer_idx -> complexity vector
        - magnitudes: dict mapping layer_idx -> L2 norm of complexity vector
    """
    print(f"\n{'='*80}")
    print(f"Computing complexity vectors")
    print(f"{'='*80}\n")

    num_layers = len(simple_activations)
    complexity_vectors = {}
    magnitudes = {}

    for layer_idx in tqdm(range(num_layers), desc="Computing complexity vectors"):
        # Mean across samples
        mean_simple = simple_activations[layer_idx].mean(dim=0)
        mean_regular = regular_activations[layer_idx].mean(dim=0)

        # Complexity vector (regular - simple)
        complexity_vector = mean_regular - mean_simple
        complexity_vectors[layer_idx] = complexity_vector

        # L2 norm
        magnitude = torch.norm(complexity_vector, p=2).item()
        magnitudes[layer_idx] = magnitude

    print(f"✓ Computed complexity vectors for {num_layers} layers")
    print(f"\nComplexity vector magnitudes by layer:")
    print(f"  - Min: {min(magnitudes.values()):.4f} (layer {min(magnitudes, key=magnitudes.get)})")
    print(f"  - Max: {max(magnitudes.values()):.4f} (layer {max(magnitudes, key=magnitudes.get)})")
    print(f"  - Mean: {np.mean(list(magnitudes.values())):.4f}")

    return complexity_vectors, magnitudes


def save_results(complexity_vectors: Dict[int, torch.Tensor], magnitudes: Dict[int, float]):
    """Save complexity vectors and metadata."""
    print(f"\n{'='*80}")
    print(f"Saving results")
    print(f"{'='*80}\n")

    # Save complexity vectors with distinct filename
    vectors_path = OUTPUT_DIR / "complexity_vectors_last_token.pt"
    torch.save(complexity_vectors, vectors_path)
    print(f"✓ Saved complexity vectors to {vectors_path}")

    # Save magnitudes as JSON
    magnitudes_path = OUTPUT_DIR / "complexity_magnitudes_last_token.json"
    with open(magnitudes_path, 'w') as f:
        json.dump({str(k): v for k, v in magnitudes.items()}, f, indent=2)
    print(f"✓ Saved magnitudes to {magnitudes_path}")

    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "num_pairs": NUM_PAIRS_TO_USE,
        "data_source": "Wikipedia pairs (Simple vs Regular)",
        "extraction_method": "last_token_only",
        "num_layers": len(magnitudes),
        "device": DEVICE,
        "seed": SEED,
    }
    metadata_path = OUTPUT_DIR / "complexity_metadata_last_token.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")


def visualize_results(magnitudes: Dict[int, float]):
    """Create visualization of complexity vector magnitudes across layers."""
    print(f"\n{'='*80}")
    print(f"Creating visualization")
    print(f"{'='*80}\n")

    layers = list(magnitudes.keys())
    mags = list(magnitudes.values())

    plt.figure(figsize=(12, 6))
    plt.plot(layers, mags, marker='o', linewidth=2, markersize=4, color='#E63946')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Complexity Vector Magnitude (L2 norm)', fontsize=12)
    plt.title('Complexity Vector Magnitude Across Layers\n(Last Token Extraction Method)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Highlight the peak
    max_layer = max(magnitudes, key=magnitudes.get)
    max_mag = magnitudes[max_layer]
    plt.axvline(max_layer, color='red', linestyle='--', alpha=0.5,
                label=f'Peak at layer {max_layer}')
    plt.legend()

    # Save plot
    plot_path = OUTPUT_DIR / "complexity_vector_magnitude_last_token.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {plot_path}")

    plt.close()


def main():
    """Run the complexity vector extraction experiment."""
    print("\n" + "="*80)
    print("COMPLEXITY VECTOR EXTRACTION - LAST TOKEN METHOD")
    print("="*80)

    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup
    setup_output_dir()
    model, tokenizer, num_layers = load_model_and_tokenizer()

    # Load Wikipedia pairs
    pairs = load_wikipedia_pairs()

    # Extract texts
    simple_texts = [pair['simple_text'] for pair in pairs]
    regular_texts = [pair['regular_text'] for pair in pairs]

    # Capture activations (LAST TOKEN ONLY)
    simple_activations = capture_activations_last_token(model, tokenizer, simple_texts, "simple")
    regular_activations = capture_activations_last_token(model, tokenizer, regular_texts, "regular")

    # Compute complexity vectors
    complexity_vectors, magnitudes = compute_complexity_vectors(simple_activations, regular_activations)

    # Save and visualize
    save_results(complexity_vectors, magnitudes)
    visualize_results(magnitudes)

    # Print comparison hint
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Complexity vectors: complexity_vectors_last_token.pt")
    print(f"  - Magnitudes: complexity_magnitudes_last_token.json")
    print(f"  - Visualization: complexity_vector_magnitude_last_token.png")
    print(f"  - Metadata: complexity_metadata_last_token.json")

    print(f"\n📊 Peak magnitude: Layer {max(magnitudes, key=magnitudes.get)}")
    print(f"   Compare to sequence-averaged method (typically layer 35)")

    print("\n")


if __name__ == "__main__":
    main()
