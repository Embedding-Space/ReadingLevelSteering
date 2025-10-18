"""
Tedium Vector Experiment

Explores whether repetitive/boring content creates a characteristic "tedium vector"
in model activation space, following the persona vectors methodology.

Experimental design:
- Model: Qwen3-4B-Instruct-2507 (36 layers, bf16)
- 20 interesting samples from The Pile (4096 tokens each)
- 20 boring samples (synthetic repetition, 4096 tokens each)
- Capture activations at each layer
- Compute tedium_vector = mean(boring) - mean(interesting)
- Analyze magnitude across layers
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
from tqdm import tqdm


# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 20
TARGET_TOKENS = 4096
OUTPUT_DIR = Path("./output")
SEED = 42

# Boring text templates - we'll vary these across samples
BORING_TEMPLATES = [
    "All work and no play makes Jack a dull boy. ",
    "The quick brown fox jumps over the lazy dog. ",
    "Lorem ipsum dolor sit amet consectetur adipiscing elit. ",
    "This is a test of the emergency broadcast system. ",
    "Please hold while your call is being transferred. ",
    "Your call is important to us. Please continue to hold. ",
    "The meeting has been rescheduled to a later time. ",
    "We are currently experiencing higher than normal call volumes. ",
    "Press one for English. Press two for Spanish. ",
    "Your password has been reset. Please check your email. ",
]


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

    # Load model with bf16 if available
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map=DEVICE,
    )

    model.eval()  # Set to evaluation mode

    # Get number of layers
    num_layers = len(model.model.layers)

    print(f"✓ Model loaded successfully")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Hidden dimension: {model.config.hidden_size}")
    print(f"  - Vocabulary size: {model.config.vocab_size}")
    print(f"  - Model dtype: {model.dtype}")

    return model, tokenizer, num_layers


def load_interesting_samples(tokenizer, num_samples: int, target_tokens: int) -> List[str]:
    """Load interesting/high-entropy samples from The Pile."""
    print(f"\n{'='*80}")
    print(f"Loading {num_samples} interesting samples from The Pile")
    print(f"Target length: {target_tokens} tokens per sample")
    print(f"{'='*80}\n")

    # Load The Pile dataset
    # Note: Using a subset since full Pile is huge
    print("Fetching dataset from HuggingFace...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    samples = []
    iterator = iter(dataset)

    with tqdm(total=num_samples, desc="Collecting interesting samples") as pbar:
        while len(samples) < num_samples:
            try:
                item = next(iterator)
                text = item['text']

                # Tokenize to check length
                tokens = tokenizer.encode(text, add_special_tokens=False)

                # If text is long enough, truncate to target length
                if len(tokens) >= target_tokens:
                    truncated_tokens = tokens[:target_tokens]
                    truncated_text = tokenizer.decode(truncated_tokens)
                    samples.append(truncated_text)
                    pbar.update(1)

            except StopIteration:
                print(f"⚠ Dataset exhausted, only got {len(samples)} samples")
                break

    print(f"✓ Collected {len(samples)} interesting samples")
    return samples


def generate_boring_samples(tokenizer, num_samples: int, target_tokens: int) -> List[str]:
    """Generate boring/repetitive samples."""
    print(f"\n{'='*80}")
    print(f"Generating {num_samples} boring samples")
    print(f"Target length: {target_tokens} tokens per sample")
    print(f"{'='*80}\n")

    samples = []

    with tqdm(total=num_samples, desc="Generating boring samples") as pbar:
        for i in range(num_samples):
            # Cycle through templates
            template = BORING_TEMPLATES[i % len(BORING_TEMPLATES)]

            # Repeat until we have enough tokens
            repeated_text = ""
            while True:
                repeated_text += template
                tokens = tokenizer.encode(repeated_text, add_special_tokens=False)
                if len(tokens) >= target_tokens:
                    # Truncate to exact length
                    truncated_tokens = tokens[:target_tokens]
                    truncated_text = tokenizer.decode(truncated_tokens)
                    samples.append(truncated_text)
                    break

            pbar.update(1)

    print(f"✓ Generated {len(samples)} boring samples")
    print(f"  - Using {len(BORING_TEMPLATES)} different repetition templates")
    return samples


def capture_activations(model, tokenizer, texts: List[str], label: str) -> Dict[int, torch.Tensor]:
    """
    Capture activations from all layers for a list of texts.

    Returns dict mapping layer_idx -> tensor of shape [num_samples, hidden_dim]
    (averaged over sequence length)
    """
    print(f"\n{'='*80}")
    print(f"Capturing activations for {len(texts)} {label} samples")
    print(f"{'='*80}\n")

    num_layers = len(model.model.layers)
    layer_activations = {i: [] for i in range(num_layers)}

    with torch.no_grad():
        for idx, text in enumerate(tqdm(texts, desc=f"Processing {label} samples")):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=TARGET_TOKENS)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Register hooks to capture layer outputs
            activations = {}

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output is the hidden states after this layer
                    # Shape: [batch_size, seq_len, hidden_dim]
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

            # Average activations over sequence length for each layer
            for layer_idx in range(num_layers):
                # Shape: [batch_size, seq_len, hidden_dim] -> [hidden_dim]
                mean_activation = activations[layer_idx].mean(dim=1).squeeze(0)
                layer_activations[layer_idx].append(mean_activation.cpu())

    # Stack activations for each layer
    # Convert list of tensors -> single tensor [num_samples, hidden_dim]
    stacked_activations = {}
    for layer_idx in range(num_layers):
        stacked_activations[layer_idx] = torch.stack(layer_activations[layer_idx])

    print(f"✓ Captured activations from {num_layers} layers")
    print(f"  - Activation shape per layer: {stacked_activations[0].shape}")

    return stacked_activations


def compute_tedium_vectors(
    interesting_activations: Dict[int, torch.Tensor],
    boring_activations: Dict[int, torch.Tensor]
) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
    """
    Compute tedium vectors and their magnitudes.

    tedium_vector = mean(boring) - mean(interesting)

    Returns:
        - tedium_vectors: dict mapping layer_idx -> tedium vector
        - magnitudes: dict mapping layer_idx -> L2 norm of tedium vector
    """
    print(f"\n{'='*80}")
    print(f"Computing tedium vectors")
    print(f"{'='*80}\n")

    num_layers = len(interesting_activations)
    tedium_vectors = {}
    magnitudes = {}

    for layer_idx in tqdm(range(num_layers), desc="Computing tedium vectors"):
        # Mean across samples: [num_samples, hidden_dim] -> [hidden_dim]
        mean_interesting = interesting_activations[layer_idx].mean(dim=0)
        mean_boring = boring_activations[layer_idx].mean(dim=0)

        # Tedium vector
        tedium_vector = mean_boring - mean_interesting
        tedium_vectors[layer_idx] = tedium_vector

        # L2 norm
        magnitude = torch.norm(tedium_vector, p=2).item()
        magnitudes[layer_idx] = magnitude

    print(f"✓ Computed tedium vectors for {num_layers} layers")
    print(f"\nTedium vector magnitudes by layer:")
    print(f"  - Min: {min(magnitudes.values()):.4f} (layer {min(magnitudes, key=magnitudes.get)})")
    print(f"  - Max: {max(magnitudes.values()):.4f} (layer {max(magnitudes, key=magnitudes.get)})")
    print(f"  - Mean: {np.mean(list(magnitudes.values())):.4f}")

    return tedium_vectors, magnitudes


def save_results(tedium_vectors: Dict[int, torch.Tensor], magnitudes: Dict[int, float]):
    """Save tedium vectors and metadata."""
    print(f"\n{'='*80}")
    print(f"Saving results")
    print(f"{'='*80}\n")

    # Save tedium vectors
    vectors_path = OUTPUT_DIR / "tedium_vectors.pt"
    torch.save(tedium_vectors, vectors_path)
    print(f"✓ Saved tedium vectors to {vectors_path}")

    # Save magnitudes as JSON
    magnitudes_path = OUTPUT_DIR / "magnitudes.json"
    with open(magnitudes_path, 'w') as f:
        json.dump({str(k): v for k, v in magnitudes.items()}, f, indent=2)
    print(f"✓ Saved magnitudes to {magnitudes_path}")

    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "num_samples": NUM_SAMPLES,
        "target_tokens": TARGET_TOKENS,
        "num_layers": len(magnitudes),
        "device": DEVICE,
        "seed": SEED,
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")


def visualize_results(magnitudes: Dict[int, float]):
    """Create visualization of tedium vector magnitudes across layers."""
    print(f"\n{'='*80}")
    print(f"Creating visualization")
    print(f"{'='*80}\n")

    layers = list(magnitudes.keys())
    mags = list(magnitudes.values())

    plt.figure(figsize=(12, 6))
    plt.plot(layers, mags, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Tedium Vector Magnitude (L2 norm)', fontsize=12)
    plt.title('Tedium Vector Magnitude Across Layers', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Highlight the peak
    max_layer = max(magnitudes, key=magnitudes.get)
    max_mag = magnitudes[max_layer]
    plt.axvline(max_layer, color='red', linestyle='--', alpha=0.5, label=f'Peak at layer {max_layer}')
    plt.legend()

    # Save plot
    plot_path = OUTPUT_DIR / "tedium_vector_magnitude.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {plot_path}")

    plt.close()


def main():
    """Run the full tedium vector experiment."""
    print("\n" + "="*80)
    print("TEDIUM VECTOR EXPERIMENT")
    print("="*80)

    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup
    setup_output_dir()
    model, tokenizer, num_layers = load_model_and_tokenizer()

    # Load data
    interesting_samples = load_interesting_samples(tokenizer, NUM_SAMPLES, TARGET_TOKENS)
    boring_samples = generate_boring_samples(tokenizer, NUM_SAMPLES, TARGET_TOKENS)

    # Capture activations
    interesting_activations = capture_activations(model, tokenizer, interesting_samples, "interesting")
    boring_activations = capture_activations(model, tokenizer, boring_samples, "boring")

    # Compute tedium vectors
    tedium_vectors, magnitudes = compute_tedium_vectors(interesting_activations, boring_activations)

    # Save and visualize
    save_results(tedium_vectors, magnitudes)
    visualize_results(magnitudes)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Tedium vectors: tedium_vectors.pt")
    print(f"  - Magnitudes: magnitudes.json")
    print(f"  - Visualization: tedium_vector_magnitude.png")
    print(f"  - Metadata: metadata.json")
    print("\n")


if __name__ == "__main__":
    main()
