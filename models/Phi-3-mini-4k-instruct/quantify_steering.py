"""
Quantitative Analysis of Steering Vector Effects

Measures the relationship between steering strength and text complexity metrics,
particularly reading level (Flesch-Kincaid grade level) and sentence structure.

Experimental design:
- Single prompt to isolate steering effect
- Range: -1.0 to +1.0 in 0.1 increments (21 data points)
- Metrics: Flesch-Kincaid grade, sentence length, word complexity
- Output: CSV data + matplotlib visualization
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import textstat
from typing import Dict


# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
VECTORS_PATH = Path("./output/complexity_vectors.pt")
# Phi-3-mini has 32 layers (0-31), use the final layer
STEERING_LAYER = 31

# Test configuration
# TEST_PROMPT = "Tell me about the history of the Internet."
TEST_PROMPT = "Can you explain quantum mechanics please?"
STRENGTH_MIN = -2.0
STRENGTH_MAX = 1.75
STRENGTH_STEP = 0.1

# Generation parameters
MAX_NEW_TOKENS = 200
# Using greedy decoding (do_sample=False) for deterministic, reproducible results

# Output
OUTPUT_DIR = Path("./output")


def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print(f"\n{'='*80}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map=DEVICE,
    )
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, tokenizer


def load_complexity_vector() -> torch.Tensor:
    """Load the complexity vector for the target layer."""
    print(f"\n{'='*80}")
    print(f"Loading complexity vector from layer {STEERING_LAYER}")
    print(f"{'='*80}\n")

    vectors = torch.load(VECTORS_PATH)
    complexity_vector = vectors[STEERING_LAYER]

    print(f"✓ Loaded vector with dimension {complexity_vector.shape[0]}")
    return complexity_vector


def get_model_layers(model):
    """Get the layers from the model, handling different architectures."""
    # Phi-3 uses standard model.model.layers structure
    return model.model.layers


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    complexity_vector: torch.Tensor,
    steering_strength: float,
) -> str:
    """Generate text with specified steering strength."""
    # Format prompt using chat template for instruct model
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if steering_strength == 0.0:
        # Baseline - no steering (greedy decoding)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=32000,  # Only stop on <|endoftext|>, not <|end|> or <|assistant|>
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply steering
    steering_vector = complexity_vector.to(model.device) * steering_strength

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        steered = hidden_states + steering_vector.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        else:
            return steered

    layers = get_model_layers(model)
    hook_handle = layers[STEERING_LAYER].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=32000,  # Only stop on <|endoftext|>, not <|end|> or <|assistant|>
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        hook_handle.remove()

    return generated_text


def analyze_text(text: str) -> Dict[str, float]:
    """
    Compute readability and complexity metrics for generated text.

    Returns dict with:
    - flesch_kincaid_grade: Grade level (e.g., 8.2 = 8th grade)
    - flesch_reading_ease: 0-100 scale (higher = easier)
    - avg_sentence_length: Average words per sentence
    - avg_syllables_per_word: Average syllables per word
    - sentence_count: Number of sentences
    - word_count: Total words
    """
    # Handle empty or very short text
    if not text or len(text.strip()) < 10:
        return {
            'flesch_kincaid_grade': 0.0,
            'flesch_reading_ease': 0.0,
            'avg_sentence_length': 0.0,
            'avg_syllables_per_word': 0.0,
            'sentence_count': 0,
            'word_count': 0,
        }

    try:
        return {
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
            'avg_syllables_per_word': textstat.avg_syllables_per_word(text),
            'sentence_count': textstat.sentence_count(text),
            'word_count': textstat.lexicon_count(text, removepunct=True),
        }
    except:
        # Fallback for edge cases
        return {
            'flesch_kincaid_grade': 0.0,
            'flesch_reading_ease': 0.0,
            'avg_sentence_length': 0.0,
            'avg_syllables_per_word': 0.0,
            'sentence_count': 0,
            'word_count': 0,
        }


def run_experiment(model, tokenizer, complexity_vector):
    """Run steering experiment across range of strengths."""
    print(f"\n{'='*80}")
    print(f"Running Quantitative Steering Analysis (Complexity Vector)")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  - Prompt: {TEST_PROMPT}")
    print(f"  - Strength range: {STRENGTH_MIN} to {STRENGTH_MAX} by {STRENGTH_STEP}")
    print(f"  - Target layer: {STEERING_LAYER}")
    print(f"  - Max tokens: {MAX_NEW_TOKENS}")
    print(f"\n")

    # Generate strengths to test
    import numpy as np
    strengths = np.arange(STRENGTH_MIN, STRENGTH_MAX + STRENGTH_STEP/2, STRENGTH_STEP)

    results = []

    for strength in strengths:
        print(f"Testing strength {strength:+.1f}...")

        # Generate text
        full_text = generate_with_steering(model, tokenizer, TEST_PROMPT, complexity_vector, strength)

        # Extract only the assistant's response (after chat template formatting)
        # Phi-3 uses <|assistant|>\n format
        if "<|assistant|>\n" in full_text:
            # Phi-3 format
            generated_text = full_text.split("<|assistant|>\n", 1)[1]
        elif "<|im_start|>assistant\n" in full_text:
            # Qwen format (fallback)
            generated_text = full_text.split("<|im_start|>assistant\n", 1)[1]
        elif "assistant\n" in full_text:
            # Generic fallback
            generated_text = full_text.split("assistant\n", 1)[1]
        else:
            # Last resort: just remove the original prompt
            generated_text = full_text[len(TEST_PROMPT):].strip()

        # Clean up any trailing special tokens
        for token in ["<|end|>", "<eos>", "<|im_end|>", "<|endoftext|>"]:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]

        generated_text = generated_text.strip()

        # Analyze
        metrics = analyze_text(generated_text)

        # Store results
        result = {
            'steering_strength': strength,
            'generated_text': generated_text,
            **metrics
        }
        results.append(result)

        # Print summary
        print(f"  Grade level: {metrics['flesch_kincaid_grade']:.1f}, "
              f"Avg sentence length: {metrics['avg_sentence_length']:.1f} words, "
              f"Word count: {metrics['word_count']}")

    return pd.DataFrame(results)


def visualize_results(df: pd.DataFrame):
    """Create visualizations of steering effects."""
    print(f"\n{'='*80}")
    print(f"Creating Visualizations")
    print(f"{'='*80}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Steering Vector Effects on Text Complexity', fontsize=16, fontweight='bold')

    # Plot 1: Flesch-Kincaid Grade Level
    ax1 = axes[0, 0]
    ax1.plot(df['steering_strength'], df['flesch_kincaid_grade'],
             marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax1.axvline(0, color='red', linestyle='--', alpha=0.3, label='Baseline (no steering)')
    ax1.set_xlabel('Steering Strength', fontsize=11)
    ax1.set_ylabel('Flesch-Kincaid Grade Level', fontsize=11)
    ax1.set_title('Reading Level vs Steering Strength')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Average Sentence Length
    ax2 = axes[0, 1]
    ax2.plot(df['steering_strength'], df['avg_sentence_length'],
             marker='s', linewidth=2, markersize=6, color='#A23B72')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.3, label='Baseline (no steering)')
    ax2.set_xlabel('Steering Strength', fontsize=11)
    ax2.set_ylabel('Average Words per Sentence', fontsize=11)
    ax2.set_title('Sentence Length vs Steering Strength')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Flesch Reading Ease
    ax3 = axes[1, 0]
    ax3.plot(df['steering_strength'], df['flesch_reading_ease'],
             marker='d', linewidth=2, markersize=6, color='#F18F01')
    ax3.axvline(0, color='red', linestyle='--', alpha=0.3, label='Baseline (no steering)')
    ax3.set_xlabel('Steering Strength', fontsize=11)
    ax3.set_ylabel('Flesch Reading Ease Score', fontsize=11)
    ax3.set_title('Reading Ease vs Steering Strength\n(Higher = Easier)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Total Word Count
    ax4 = axes[1, 1]
    ax4.plot(df['steering_strength'], df['word_count'],
             marker='^', linewidth=2, markersize=6, color='#6A994E')
    ax4.axvline(0, color='red', linestyle='--', alpha=0.3, label='Baseline (no steering)')
    ax4.set_xlabel('Steering Strength', fontsize=11)
    ax4.set_ylabel('Total Words Generated', fontsize=11)
    ax4.set_title('Output Length vs Steering Strength')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # Save
    plot_path = OUTPUT_DIR / "steering_quantitative_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {plot_path}")

    plt.close()


def main():
    """Run quantitative steering analysis."""
    print("\n" + "="*80)
    print("QUANTITATIVE STEERING ANALYSIS")
    print("="*80)

    # Load model and vectors
    model, tokenizer = load_model_and_tokenizer()
    complexity_vector = load_complexity_vector()

    # Run experiment
    df = run_experiment(model, tokenizer, complexity_vector)

    # Save results
    csv_path = OUTPUT_DIR / "steering_quantitative_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")

    # Create visualizations
    visualize_results(df)

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    print("Flesch-Kincaid Grade Level:")
    print(f"  Range: {df['flesch_kincaid_grade'].min():.1f} to {df['flesch_kincaid_grade'].max():.1f}")
    baseline_row = df.iloc[(df['steering_strength'] - 0.0).abs().argsort()[0]]
    print(f"  Baseline (~0.0): {baseline_row['flesch_kincaid_grade']:.1f}")

    print("\nAverage Sentence Length:")
    print(f"  Range: {df['avg_sentence_length'].min():.1f} to {df['avg_sentence_length'].max():.1f} words")
    print(f"  Baseline (~0.0): {baseline_row['avg_sentence_length']:.1f} words")

    print("\nTotal Output Length:")
    print(f"  Range: {df['word_count'].min()} to {df['word_count'].max()} words")
    print(f"  Baseline (~0.0): {baseline_row['word_count']} words")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Plot: {OUTPUT_DIR / 'steering_quantitative_analysis.png'}")
    print("\n")


if __name__ == "__main__":
    main()
