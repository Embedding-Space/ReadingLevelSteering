"""
Activation Matrix Visualization for Blog Post

Visualizes raw activation matrices from Wikipedia pair processing to show
what the "X-rays" of model activations actually look like.

Creates two plots:
1. 2×3 grid: Simple/Regular × Layer 0/17/35 raw activations
2. 1×3 grid: Difference matrices (Regular - Simple) at each layer

Each matrix is 4096 tokens (horizontal) × 2560 features (vertical),
displayed with blue-white-red colormap (values clamped to [-2, 2]).
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict
from tqdm import tqdm


# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path("../../data/wikipedia_pairs/wikipedia_pairs.json")
OUTPUT_DIR = Path("./output")
SEED = 42

# Article to visualize (easily changeable)
ARTICLE_TOPIC = "World War II"

# Layers to visualize (input, middle, output)
LAYERS_TO_VISUALIZE = [0, 17, 35]

# Visualization settings
CLAMP_MIN = -2.0
CLAMP_MAX = 2.0
COLORMAP = "RdBu_r"  # Red-white-blue (reversed so blue=negative, red=positive)


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


def load_article_pair() -> Dict:
    """Load the specified article pair."""
    print(f"\n{'='*80}")
    print(f"Loading article pair: {ARTICLE_TOPIC}")
    print(f"{'='*80}\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Wikipedia pairs not found at {DATA_PATH}. "
            "Run prepare_wikipedia_pairs.py first!"
        )

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find the article pair by topic
    pair = None
    for p in data['pairs']:
        if p['topic'] == ARTICLE_TOPIC:
            pair = p
            break

    if pair is None:
        available_topics = [p['topic'] for p in data['pairs']]
        raise ValueError(
            f"Article '{ARTICLE_TOPIC}' not found. Available topics:\n"
            + "\n".join(f"  - {topic}" for topic in available_topics)
        )

    print(f"✓ Found article pair: {pair['topic']}")
    print(f"  - Simple grade level: {pair['simple_grade']:.1f}")
    print(f"  - Regular grade level: {pair['regular_grade']:.1f}")

    return pair


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


def plot_raw_matrices(simple_matrices: Dict[int, torch.Tensor],
                      regular_matrices: Dict[int, torch.Tensor],
                      article_topic: str):
    """
    Create 2×3 grid plot of raw activation matrices.

    Rows: Simple, Regular
    Columns: Layer 0, 17, 35
    """
    print(f"\n{'='*80}")
    print(f"Creating raw activation matrices plot")
    print(f"{'='*80}\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Activation Matrices: {article_topic}', fontsize=16, fontweight='bold')

    layers = sorted(simple_matrices.keys())
    row_labels = ['Simple Wikipedia', 'Regular Wikipedia']

    for row_idx, (matrices, label) in enumerate([(simple_matrices, 'Simple'),
                                                   (regular_matrices, 'Regular')]):
        for col_idx, layer_idx in enumerate(layers):
            ax = axes[row_idx, col_idx]
            matrix = matrices[layer_idx].numpy()

            # Transpose to get landscape orientation: tokens horizontal, features vertical
            matrix = matrix.T  # Now shape: [hidden_dim=2560, seq_len=4096]

            # Clamp values
            matrix_clamped = np.clip(matrix, CLAMP_MIN, CLAMP_MAX)

            # Plot
            im = ax.imshow(matrix_clamped, cmap=COLORMAP, aspect='auto',
                          vmin=CLAMP_MIN, vmax=CLAMP_MAX, interpolation='nearest')

            # Labels
            ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Features (2560)' if col_idx == 0 else '', fontsize=10)
            ax.set_xlabel('Tokens (4096)' if row_idx == 1 else '', fontsize=10)

            # Add row label on the left
            if col_idx == 0:
                ax.text(-0.15, 0.5, row_labels[row_idx], transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', rotation=90)

            # Reduce tick density
            ax.set_xticks([0, 1024, 2048, 3072, 4096])
            ax.set_yticks([0, 640, 1280, 1920, 2560])

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / f'activation_matrices_raw_{article_topic.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved raw matrices plot: {output_path}")

    plt.close()


def plot_difference_matrices(simple_matrices: Dict[int, torch.Tensor],
                             regular_matrices: Dict[int, torch.Tensor],
                             article_topic: str):
    """
    Create 1×3 grid plot of difference matrices (Regular - Simple).

    Columns: Layer 0, 17, 35
    """
    print(f"\n{'='*80}")
    print(f"Creating difference matrices plot")
    print(f"{'='*80}\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Activation Differences (Regular - Simple): {article_topic}',
                fontsize=16, fontweight='bold')

    layers = sorted(simple_matrices.keys())

    for col_idx, layer_idx in enumerate(layers):
        ax = axes[col_idx]

        simple_matrix = simple_matrices[layer_idx].numpy().T  # Transpose for landscape
        regular_matrix = regular_matrices[layer_idx].numpy().T

        # Compute difference
        diff_matrix = regular_matrix - simple_matrix

        # Clamp values
        diff_clamped = np.clip(diff_matrix, CLAMP_MIN, CLAMP_MAX)

        # Plot
        im = ax.imshow(diff_clamped, cmap=COLORMAP, aspect='auto',
                      vmin=CLAMP_MIN, vmax=CLAMP_MAX, interpolation='nearest')

        # Labels
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features (2560)' if col_idx == 0 else '', fontsize=10)
        ax.set_xlabel('Tokens (4096)', fontsize=10)

        # Reduce tick density
        ax.set_xticks([0, 1024, 2048, 3072, 4096])
        ax.set_yticks([0, 640, 1280, 1920, 2560])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = OUTPUT_DIR / f'activation_matrices_diff_{article_topic.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved difference matrices plot: {output_path}")

    plt.close()


def main():
    """Main execution."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    setup_output_dir()

    # Load model and data
    model, tokenizer, num_layers = load_model_and_tokenizer()
    pair = load_article_pair()

    # Verify requested layers exist
    for layer_idx in LAYERS_TO_VISUALIZE:
        if layer_idx >= num_layers:
            raise ValueError(f"Layer {layer_idx} requested but model only has {num_layers} layers")

    # Capture activation matrices for both texts
    print(f"\n{'='*80}")
    print(f"Processing Simple Wikipedia text")
    print(f"{'='*80}")
    simple_matrices = capture_activation_matrix(model, tokenizer, pair['simple_text'], LAYERS_TO_VISUALIZE)

    print(f"\n{'='*80}")
    print(f"Processing Regular Wikipedia text")
    print(f"{'='*80}")
    regular_matrices = capture_activation_matrix(model, tokenizer, pair['regular_text'], LAYERS_TO_VISUALIZE)

    # Create visualizations
    plot_raw_matrices(simple_matrices, regular_matrices, pair['topic'])
    plot_difference_matrices(simple_matrices, regular_matrices, pair['topic'])

    print(f"\n{'='*80}")
    print(f"✓ Visualization complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
