"""
Tedium Vector Steering Experiment

Tests whether the extracted "tedium vector" can steer model output toward/away from
complexity, verbosity, or simplicity.

Experimental design:
- Load pre-extracted tedium vectors
- Generate text with three conditions:
  1. Baseline (no steering)
  2. Steer TOWARD tedium (add vector to activations)
  3. Steer AWAY from tedium (subtract vector from activations)
- Compare outputs qualitatively to understand what the vector captures
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import argparse
from typing import Dict, Optional


# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
VECTORS_PATH = Path("./output/tedium_vectors.pt")
METADATA_PATH = Path("./output/metadata.json")

# Steering configuration
STEERING_LAYER = 35  # Layer with strongest signal (from extraction experiment)
DEFAULT_STEERING_STRENGTH = 1.0  # Default multiplier for vector addition/subtraction

# Generation parameters
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# Test prompts - varied to explore different aspects of the vector
TEST_PROMPTS = [
    "Explain how photosynthesis works.",
    "Write a short story about a robot learning to paint.",
    "Describe the process of baking bread from scratch.",
    "What are the main causes of climate change?",
    "Tell me about the history of the Internet.",
]


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

    print(f"✓ Model loaded successfully")
    print(f"  - Number of layers: {len(model.model.layers)}")
    print(f"  - Model dtype: {model.dtype}")

    return model, tokenizer


def load_tedium_vectors() -> Dict[int, torch.Tensor]:
    """Load pre-extracted tedium vectors."""
    print(f"\n{'='*80}")
    print(f"Loading tedium vectors")
    print(f"{'='*80}\n")

    if not VECTORS_PATH.exists():
        raise FileNotFoundError(
            f"Tedium vectors not found at {VECTORS_PATH}. "
            "Run extract_tedium_vectors.py first!"
        )

    vectors = torch.load(VECTORS_PATH)

    # Load metadata for context
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print(f"✓ Loaded tedium vectors from {VECTORS_PATH}")
    print(f"  - Number of layers: {len(vectors)}")
    print(f"  - Vector dimension: {vectors[0].shape[0]}")
    print(f"  - Extracted from {metadata['num_samples']} samples per condition")

    return vectors


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    tedium_vector: Optional[torch.Tensor] = None,
    steering_direction: int = 0,  # 0 = baseline, +1 = toward tedium, -1 = away from tedium
    steering_strength: float = DEFAULT_STEERING_STRENGTH,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generate text with optional steering.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        tedium_vector: The tedium vector to use for steering (single layer)
        steering_direction: 0 (baseline), +1 (toward tedium), -1 (away from tedium)
        steering_strength: Multiplier for vector addition
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if steering_direction == 0 or tedium_vector is None:
        # Baseline generation - no steering
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    else:
        # Steering generation - modify activations at target layer
        steering_vector = tedium_vector.to(model.device) * steering_direction * steering_strength

        # We need to hook into the generation process
        # This is trickier with .generate() so we'll do it manually with a forward hook

        def steering_hook(module, input, output):
            # output shape: [batch_size, seq_len, hidden_dim]
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add steering vector to all positions
            # steering_vector shape: [hidden_dim]
            # Broadcast to match hidden_states
            steered = hidden_states + steering_vector.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            else:
                return steered

        # Register hook on target layer
        hook_handle = model.model.layers[STEERING_LAYER].register_forward_hook(steering_hook)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        finally:
            # Remove hook
            hook_handle.remove()

        return generated_text


def run_steering_experiments(model, tokenizer, tedium_vectors: Dict[int, torch.Tensor], steering_strength: float):
    """Run steering experiments with all test prompts."""
    print(f"\n{'='*80}")
    print(f"Running Steering Experiments")
    print(f"{'='*80}")
    print(f"\nSteering configuration:")
    print(f"  - Target layer: {STEERING_LAYER}")
    print(f"  - Steering strength: {steering_strength}")
    print(f"  - Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Top-p: {TOP_P}")
    print(f"\n")

    # Get the tedium vector for our target layer
    tedium_vector = tedium_vectors[STEERING_LAYER]

    results = []

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─'*80}")
        print(f"Prompt {idx}/{len(TEST_PROMPTS)}: {prompt}")
        print(f"{'─'*80}\n")

        # Baseline
        print("🔵 BASELINE (no steering)")
        baseline = generate_with_steering(
            model, tokenizer, prompt,
            tedium_vector=None,
            steering_direction=0
        )
        # Extract only the generated part (after the prompt)
        baseline_output = baseline[len(prompt):].strip()
        print(f"{baseline_output}\n")

        # Toward tedium
        print(f"🟢 TOWARD TEDIUM (steering strength +{steering_strength})")
        toward = generate_with_steering(
            model, tokenizer, prompt,
            tedium_vector=tedium_vector,
            steering_direction=+1,
            steering_strength=steering_strength
        )
        toward_output = toward[len(prompt):].strip()
        print(f"{toward_output}\n")

        # Away from tedium
        print(f"🔴 AWAY FROM TEDIUM (steering strength -{steering_strength})")
        away = generate_with_steering(
            model, tokenizer, prompt,
            tedium_vector=tedium_vector,
            steering_direction=-1,
            steering_strength=steering_strength
        )
        away_output = away[len(prompt):].strip()
        print(f"{away_output}\n")

        results.append({
            "prompt": prompt,
            "baseline": baseline_output,
            "toward_tedium": toward_output,
            "away_from_tedium": away_output,
        })

    # Save results with strength in filename
    output_path = Path(f"./output/steering_results_strength_{steering_strength:.1f}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to {output_path}")
    print(f"{'='*80}\n")


def main():
    """Run the steering experiment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Steer model output using tedium vector")
    parser.add_argument(
        "--strength",
        type=float,
        default=DEFAULT_STEERING_STRENGTH,
        help=f"Steering strength multiplier (default: {DEFAULT_STEERING_STRENGTH})"
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("TEDIUM VECTOR STEERING EXPERIMENT")
    print("="*80)

    # Load model and vectors
    model, tokenizer = load_model_and_tokenizer()
    tedium_vectors = load_tedium_vectors()

    # Run experiments
    run_steering_experiments(model, tokenizer, tedium_vectors, args.strength)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nAnalyze the outputs to understand what the tedium vector captures:")
    print("  - Does 'toward tedium' produce simpler/shorter/more repetitive text?")
    print("  - Does 'away from tedium' produce more complex/verbose/varied text?")
    print("  - What does this tell us about what we actually extracted?")
    print("\n")


if __name__ == "__main__":
    main()
