"""
Prompted Control Experiment

Tests whether the model can voluntarily control reading level when explicitly
prompted, versus requiring geometric intervention via activation steering.

Experimental design:
- Prompt model to explain quantum mechanics at specific grade levels (5-20)
- Measure actual output complexity using same metrics as steering experiments
- Compare R² of (requested, actual) to R² of activation steering
- Determine if prompting alone provides sufficient control
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import textstat
from typing import Dict


# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Test configuration
BASE_PROMPT = "Can you explain quantum mechanics please?"
REQUESTED_GRADES = list(range(5, 21))  # Grades 5-20

# Generation parameters
MAX_NEW_TOKENS = 200
# Using greedy decoding for deterministic results

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
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()

    print(f"Model loaded successfully on {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")

    return model, tokenizer


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from Qwen model outputs."""
    import re
    # Remove everything between <think> and </think>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def calculate_text_statistics(text: str) -> Dict[str, float]:
    """Calculate complexity metrics for generated text."""
    # Strip thinking tags first (for Qwen models)
    text = strip_thinking_tags(text)

    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'avg_sentence_length': textstat.avg_sentence_length(text),
        'avg_syllables_per_word': textstat.avg_syllables_per_word(text),
        'word_count': len(text.split()),
    }


def generate_at_grade_level(model, tokenizer, grade_level: int) -> str:
    """Generate response with explicit grade level instruction."""
    # Construct prompt with grade level instruction
    prompt = f"Please explain quantum mechanics at a {grade_level}th grade reading level."

    # Format with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy decoding
        )

    # Decode and extract just the assistant's response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response after the assistant marker
    if "<|im_start|>assistant" in full_text:
        response = full_text.split("<|im_start|>assistant")[-1].strip()
    else:
        response = full_text

    return response


def run_prompted_control_experiment():
    """Run the full prompted control experiment."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Collect results
    results = []

    print(f"\n{'='*80}")
    print("PROMPTED CONTROL EXPERIMENT")
    print(f"{'='*80}\n")
    print(f"Testing requested grade levels: {REQUESTED_GRADES[0]} to {REQUESTED_GRADES[-1]}")
    print(f"Total samples: {len(REQUESTED_GRADES)}\n")

    for grade in REQUESTED_GRADES:
        print(f"[Grade {grade:2d}] Generating...", end=" ", flush=True)

        # Generate response
        response = generate_at_grade_level(model, tokenizer, grade)

        # Calculate metrics
        stats = calculate_text_statistics(response)

        print(f"Actual FK grade: {stats['flesch_kincaid_grade']:.2f}", flush=True)

        # Record results
        results.append({
            'requested_grade': grade,
            'actual_grade': stats['flesch_kincaid_grade'],
            'flesch_reading_ease': stats['flesch_reading_ease'],
            'avg_sentence_length': stats['avg_sentence_length'],
            'avg_syllables_per_word': stats['avg_syllables_per_word'],
            'word_count': stats['word_count'],
            'response': response,
        })

    # Save results
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "prompted_control_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Calculate correlation statistics
    from scipy import stats as scipy_stats

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        df['requested_grade'], df['actual_grade']
    )
    r_squared = r_value ** 2

    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    print(f"Linear regression: actual_grade = {slope:.4f} × requested_grade + {intercept:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"p-value = {p_value:.2e}")
    print(f"Standard error = {std_err:.4f}")
    print(f"\nInterpretation:")
    if r_squared > 0.8:
        print(f"  → STRONG correlation - prompting provides reliable control")
        print(f"  → Slope = {slope:.2f} (ideal would be 1.0)")
    elif r_squared > 0.5:
        print(f"  → MODERATE correlation - prompting has some effect")
        print(f"  → But activation steering (R²≈0.92) provides superior control")
    else:
        print(f"  → WEAK correlation - prompting does not reliably control grade level")
        print(f"  → Activation steering (R²≈0.92) is dramatically superior")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Requested vs Actual Grade Level
    ax1.scatter(df['requested_grade'], df['actual_grade'],
                color='#2E86AB', s=80, alpha=0.7, edgecolors='black', linewidth=1)

    # Add regression line
    x_line = [REQUESTED_GRADES[0], REQUESTED_GRADES[-1]]
    y_line = [slope * x + intercept for x in x_line]
    ax1.plot(x_line, y_line, 'r--', linewidth=2, label=f'Linear fit (R²={r_squared:.3f})')

    # Add ideal line (slope=1)
    ax1.plot(x_line, x_line, 'g:', linewidth=2, label='Ideal (slope=1.0)', alpha=0.5)

    ax1.set_xlabel('Requested Grade Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual Flesch-Kincaid Grade', fontsize=12, fontweight='bold')
    ax1.set_title('Prompted Control: Can Model Follow Grade Level Instructions?',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Add statistics text box
    stats_text = (
        f'Slope: {slope:.3f}\n'
        f'R²: {r_squared:.4f}\n'
        f'p-value: {p_value:.2e}'
    )
    ax1.text(0.05, 0.95, stats_text,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Residuals (how far off from requested)
    residuals = df['actual_grade'] - df['requested_grade']
    ax2.scatter(df['requested_grade'], residuals,
                color='#F18F01', s=80, alpha=0.7, edgecolors='black', linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect accuracy')

    ax2.set_xlabel('Requested Grade Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual (Actual - Requested)', fontsize=12, fontweight='bold')
    ax2.set_title('Deviation from Requested Grade Level',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    # Add mean residual
    mean_residual = residuals.mean()
    ax2.text(0.05, 0.95, f'Mean deviation: {mean_residual:+.2f} grades',
             transform=ax2.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "prompted_control_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")

    return df, r_squared, slope


if __name__ == "__main__":
    df, r_squared, slope = run_prompted_control_experiment()
