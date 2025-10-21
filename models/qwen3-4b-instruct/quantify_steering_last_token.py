"""
Quantitative Analysis of Steering with Last-Token V_c

Tests whether the last-token extraction method produces effective steering
despite poor validation correlation (R²=0.12).

Uses the same experimental design as the original quantify_steering.py:
- Single prompt to isolate steering effect
- Range: -5.0 to +5.0 in 0.25 increments (41 data points)
- Metrics: Flesch-Kincaid grade, reading ease, sentence structure
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
import numpy as np


# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
VECTORS_PATH = Path("./output/complexity_vectors_last_token.pt")  # Last-token version
STEERING_LAYER = 35

# Test configuration
TEST_PROMPT = "Can you explain quantum mechanics please?"
STRENGTH_MIN = -5.0
STRENGTH_MAX = 5.0
STRENGTH_STEP = 0.25

# Generation parameters
MAX_NEW_TOKENS = 200

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
    """Load the last-token complexity vector for the target layer."""
    print(f"\n{'='*80}")
    print(f"Loading last-token complexity vector from layer {STEERING_LAYER}")
    print(f"{'='*80}\n")

    vectors = torch.load(VECTORS_PATH)
    complexity_vector = vectors[STEERING_LAYER]

    print(f"✓ Loaded vector with dimension {complexity_vector.shape[0]}")
    print(f"  (Extracted using last-token method)")
    return complexity_vector


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

    hook_handle = model.model.layers[STEERING_LAYER].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
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
    except Exception as e:
        print(f"Warning: textstat error: {e}")
        return {
            'flesch_kincaid_grade': 0.0,
            'flesch_reading_ease': 0.0,
            'avg_sentence_length': 0.0,
            'avg_syllables_per_word': 0.0,
            'sentence_count': 0,
            'word_count': 0,
        }


def extract_assistant_response(text: str, prompt: str) -> str:
    """Extract just the assistant's response from the full generated text."""
    # Remove the original prompt if it's included
    if prompt in text:
        text = text.split(prompt, 1)[1]

    # Try to extract assistant portion (after last <|im_start|>assistant marker)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    # Clean up any remaining special tokens
    text = text.replace("<|im_end|>", "").strip()

    return text


def main():
    """Run the steering quantification experiment."""
    print("\n" + "="*80)
    print("STEERING QUANTIFICATION - LAST TOKEN V_C")
    print("="*80)

    # Load model and vector
    model, tokenizer = load_model_and_tokenizer()
    complexity_vector = load_complexity_vector()

    # Generate steering strengths to test
    strengths = np.arange(STRENGTH_MIN, STRENGTH_MAX + STRENGTH_STEP/2, STRENGTH_STEP)

    print(f"\n{'='*80}")
    print(f"Running steering experiment")
    print(f"{'='*80}\n")
    print(f"Prompt: \"{TEST_PROMPT}\"")
    print(f"Steering range: [{STRENGTH_MIN}, {STRENGTH_MAX}]")
    print(f"Step size: {STRENGTH_STEP}")
    print(f"Total samples: {len(strengths)}")
    print()

    # Collect results
    results = []

    for strength in strengths:
        print(f"[{len(results)+1}/{len(strengths)}] Testing α = {strength:+.2f}...", end=" ", flush=True)

        # Generate text
        full_text = generate_with_steering(model, tokenizer, TEST_PROMPT, complexity_vector, strength)
        response = extract_assistant_response(full_text, TEST_PROMPT)

        # Analyze
        metrics = analyze_text(response)

        print(f"FK = {metrics['flesch_kincaid_grade']:.1f}")

        # Store results
        results.append({
            'steering_strength': strength,
            'response_text': response,
            **metrics
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    csv_path = OUTPUT_DIR / "steering_quantitative_results_last_token.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")

    # Create visualization
    print(f"\n{'='*80}")
    print("Creating visualization")
    print(f"{'='*80}\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Flesch-Kincaid Grade Level
    ax = axes[0, 0]
    ax.plot(df['steering_strength'], df['flesch_kincaid_grade'],
            marker='o', linewidth=2, color='#E63946')
    ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5, label='12th grade')
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Flesch-Kincaid Grade Level', fontsize=12, fontweight='bold')
    ax.set_title('Reading Level vs. Steering Strength\n(Last-Token V_c)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Flesch Reading Ease
    ax = axes[0, 1]
    ax.plot(df['steering_strength'], df['flesch_reading_ease'],
            marker='s', linewidth=2, color='#06A77D')
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Flesch Reading Ease', fontsize=12, fontweight='bold')
    ax.set_title('Reading Ease vs. Steering Strength\n(Higher = Easier)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Average Sentence Length
    ax = axes[1, 0]
    ax.plot(df['steering_strength'], df['avg_sentence_length'],
            marker='^', linewidth=2, color='#F18F01')
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Sentence Length (words)', fontsize=12, fontweight='bold')
    ax.set_title('Sentence Complexity vs. Steering Strength',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Syllables per Word
    ax = axes[1, 1]
    ax.plot(df['steering_strength'], df['avg_syllables_per_word'],
            marker='D', linewidth=2, color='#9D4EDD')
    ax.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Syllables per Word', fontsize=12, fontweight='bold')
    ax.set_title('Word Complexity vs. Steering Strength',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "steering_analysis_last_token.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {plot_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    print(f"Flesch-Kincaid Grade Level:")
    print(f"  At α = {STRENGTH_MIN}: {df.iloc[0]['flesch_kincaid_grade']:.2f}")
    print(f"  At α = 0.0: {df[df['steering_strength'] == 0.0]['flesch_kincaid_grade'].values[0]:.2f}")
    print(f"  At α = {STRENGTH_MAX}: {df.iloc[-1]['flesch_kincaid_grade']:.2f}")
    print(f"  Total range: {df['flesch_kincaid_grade'].max() - df['flesch_kincaid_grade'].min():.2f} grades")

    print(f"\nFlesch Reading Ease:")
    print(f"  At α = {STRENGTH_MIN}: {df.iloc[0]['flesch_reading_ease']:.2f}")
    print(f"  At α = 0.0: {df[df['steering_strength'] == 0.0]['flesch_reading_ease'].values[0]:.2f}")
    print(f"  At α = {STRENGTH_MAX}: {df.iloc[-1]['flesch_reading_ease']:.2f}")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")

    print(f"Compare these results to the sequence-averaged method:")
    print(f"  steering_quantitative_results.csv (original)")
    print(f"  steering_quantitative_results_last_token.csv (this run)")
    print()


if __name__ == "__main__":
    main()
