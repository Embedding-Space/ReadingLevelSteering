"""
Tedium Activation Matrix Visualization

Recreates the original tedium vector experiment visualization by showing
activation matrices for one interesting sample (from The Pile) and one
boring sample (repetitive text).

This completes the blog post narrative by showing what the original
"X-rays" looked like before we switched to Wikipedia pairs.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("./output")
TARGET_TOKENS = 4096
SEED = 42

# Layers to visualize (input, middle, output)
LAYERS_TO_VISUALIZE = [0, 17, 35]

# Visualization settings
CLAMP_MIN = -2.0
CLAMP_MAX = 2.0
COLORMAP = "RdBu_r"  # Red-white-blue diverging

# Boring text template (Jack Torrance style)
BORING_TEMPLATE = "All work and no play makes Jack a dull boy. "


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


def load_interesting_sample(tokenizer) -> str:
    """Load one interesting sample from The Pile."""
    print(f"\n{'='*80}")
    print(f"Loading interesting sample from The Pile")
    print(f"Target length: {TARGET_TOKENS} tokens")
    print(f"{'='*80}\n")

    print("Fetching dataset from HuggingFace...")
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    # Get first sample that's long enough
    for item in dataset:
        text = item['text']
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) >= TARGET_TOKENS:
            # Truncate to exact length
            truncated_tokens = tokens[:TARGET_TOKENS]
            truncated_text = tokenizer.decode(truncated_tokens)
            print(f"✓ Found interesting sample (truncated to {TARGET_TOKENS} tokens)")
            print(f"  Preview: {truncated_text[:100]}...")
            return truncated_text

    raise RuntimeError("Could not find suitable sample in The Pile")


def generate_boring_sample(tokenizer) -> str:
    """Generate one boring/repetitive sample."""
    print(f"\n{'='*80}")
    print(f"Generating boring sample")
    print(f"Target length: {TARGET_TOKENS} tokens")
    print(f"{'='*80}\n")

    # Repeat template until we have enough tokens
    repeated_text = ""
    while True:
        repeated_text += BORING_TEMPLATE
        tokens = tokenizer.encode(repeated_text, add_special_tokens=False)
        if len(tokens) >= TARGET_TOKENS:
            # Truncate to exact length
            truncated_tokens = tokens[:TARGET_TOKENS]
            truncated_text = tokenizer.decode(truncated_tokens)
            print(f"✓ Generated boring sample ({TARGET_TOKENS} tokens)")
            print(f"  Template: {BORING_TEMPLATE.strip()}")
            return truncated_text


def capture_activation_matrix(model, tokenizer, text: str, layers_to_capture: list) -> Dict[int, torch.Tensor]:
    """
    Capture full activation matrices (not averaged) for specified layers.

    Returns dict mapping layer_idx -> tensor of shape [seq_len, hidden_dim]
    """
    print(f"\nCapturing activation matrices...")

    with torch.no_grad():
        # Tokenize to exactly 4096 tokens with padding
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096, padding='max_length')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        seq_len = inputs['input_ids'].shape[1]
        print(f"  - Sequence length: {seq_len} tokens (padded)")

        # Register hooks to capture layer outputs
        activations = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                activations[layer_idx] = output[0] if isinstance(output, tuple) else output
            return hook

        # Register hooks only for layers we want to visualize
        hooks = []
        for layer_idx in layers_to_capture:
            hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        # Forward pass
        _ = model(**inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Extract full matrices (shape: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim])
        # Convert from bfloat16 to float32 for numpy compatibility
        matrices = {}
        for layer_idx in layers_to_capture:
            matrices[layer_idx] = activations[layer_idx].squeeze(0).float().cpu()

    print(f"✓ Captured {len(matrices)} activation matrices")
    for layer_idx, matrix in matrices.items():
        print(f"  - Layer {layer_idx}: {matrix.shape}")

    return matrices


def plot_raw_matrices(interesting_matrices: Dict[int, torch.Tensor],
                      boring_matrices: Dict[int, torch.Tensor],
                      hidden_dim: int):
    """
    Create 2×3 grid plot of raw activation matrices.

    Rows: Interesting, Boring
    Columns: Layer 0, 17, 35
    """
    print(f"\n{'='*80}")
    print(f"Creating raw activation matrices plot")
    print(f"{'='*80}\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Activation Matrices: Tedium Experiment (Interesting vs. Boring)',
                 fontsize=16, fontweight='bold')

    layers = sorted(interesting_matrices.keys())
    row_labels = ['Interesting (The Pile)', 'Boring (Repetitive)']

    for row_idx, (matrices, label) in enumerate([(interesting_matrices, 'Interesting'),
                                                   (boring_matrices, 'Boring')]):
        for col_idx, layer_idx in enumerate(layers):
            ax = axes[row_idx, col_idx]
            matrix = matrices[layer_idx].numpy()

            # Transpose to get landscape orientation: tokens horizontal, features vertical
            matrix = matrix.T  # Now shape: [hidden_dim, seq_len]

            # Clamp values
            matrix_clamped = np.clip(matrix, CLAMP_MIN, CLAMP_MAX)

            # Plot
            im = ax.imshow(matrix_clamped, cmap=COLORMAP, aspect='auto',
                          vmin=CLAMP_MIN, vmax=CLAMP_MAX, interpolation='nearest')

            # Labels
            ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Features ({hidden_dim})' if col_idx == 0 else '', fontsize=10)
            ax.set_xlabel('Tokens (4096)' if row_idx == 1 else '', fontsize=10)

            # Add row label on the left
            if col_idx == 0:
                ax.text(-0.15, 0.5, row_labels[row_idx], transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', rotation=90)

            # Reduce tick density
            ax.set_xticks([0, 1024, 2048, 3072, 4096])
            ax.set_yticks([0, hidden_dim//4, hidden_dim//2, 3*hidden_dim//4, hidden_dim])

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / 'activation_matrices_raw_tedium.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved raw matrices plot: {output_path}")

    plt.close()


def plot_difference_matrices(interesting_matrices: Dict[int, torch.Tensor],
                             boring_matrices: Dict[int, torch.Tensor],
                             hidden_dim: int):
    """
    Create 1×3 grid plot of difference matrices (Boring - Interesting).

    This shows the tedium vector spatially.
    """
    print(f"\n{'='*80}")
    print(f"Creating difference matrices plot")
    print(f"{'='*80}\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Activation Differences (Boring - Interesting): Tedium Vector',
                fontsize=16, fontweight='bold')

    layers = sorted(interesting_matrices.keys())

    for col_idx, layer_idx in enumerate(layers):
        ax = axes[col_idx]

        interesting_matrix = interesting_matrices[layer_idx].numpy().T  # Transpose for landscape
        boring_matrix = boring_matrices[layer_idx].numpy().T

        # Compute difference (tedium vector)
        diff_matrix = boring_matrix - interesting_matrix

        # Clamp values
        diff_clamped = np.clip(diff_matrix, CLAMP_MIN, CLAMP_MAX)

        # Plot
        im = ax.imshow(diff_clamped, cmap=COLORMAP, aspect='auto',
                      vmin=CLAMP_MIN, vmax=CLAMP_MAX, interpolation='nearest')

        # Labels
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Features ({hidden_dim})' if col_idx == 0 else '', fontsize=10)
        ax.set_xlabel('Tokens (4096)', fontsize=10)

        # Reduce tick density
        ax.set_xticks([0, 1024, 2048, 3072, 4096])
        ax.set_yticks([0, hidden_dim//4, hidden_dim//2, 3*hidden_dim//4, hidden_dim])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = OUTPUT_DIR / 'activation_matrices_diff_tedium.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved difference matrices plot: {output_path}")

    plt.close()


def main():
    """Main execution."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    model, tokenizer, num_layers = load_model_and_tokenizer()

    # Verify requested layers exist
    for layer_idx in LAYERS_TO_VISUALIZE:
        if layer_idx >= num_layers:
            raise ValueError(f"Layer {layer_idx} requested but model only has {num_layers} layers")

    # Load/generate samples
    interesting_text = load_interesting_sample(tokenizer)
    boring_text = generate_boring_sample(tokenizer)

    # Capture activation matrices
    print(f"\n{'='*80}")
    print(f"Processing interesting text")
    print(f"{'='*80}")
    interesting_matrices = capture_activation_matrix(model, tokenizer, interesting_text, LAYERS_TO_VISUALIZE)

    print(f"\n{'='*80}")
    print(f"Processing boring text")
    print(f"{'='*80}")
    boring_matrices = capture_activation_matrix(model, tokenizer, boring_text, LAYERS_TO_VISUALIZE)

    # Create visualizations
    hidden_dim = model.config.hidden_size
    plot_raw_matrices(interesting_matrices, boring_matrices, hidden_dim)
    plot_difference_matrices(interesting_matrices, boring_matrices, hidden_dim)

    print(f"\n{'='*80}")
    print(f"✓ Visualization complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
