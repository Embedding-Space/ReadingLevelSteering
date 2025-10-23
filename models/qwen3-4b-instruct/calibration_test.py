"""
Grade Level Calibration Test

Tests how well our linear regression model predicts actual output grade levels.
Sweeps target grade levels from 6 to 18, computes corresponding α values,
generates text, and measures actual FK grade level to assess calibration accuracy.

Expected result: Points should cluster near the y=x diagonal (perfect calibration).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import textstat
import numpy as np

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
VECTORS_PATH = Path("./output/complexity_vectors.pt")
STEERING_LAYER = 35

# Test configuration
TEST_PROMPT = "Please tell me about the sun. Please do not use Markdown."
TARGET_GRADES = list(range(6, 19))  # 6, 7, 8, ..., 18
MAX_NEW_TOKENS = 1024

# Linear regression model from complexity vectors
REGRESSION_SLOPE = 1.28
REGRESSION_INTERCEPT = 11.9

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


def grade_to_alpha(target_grade: float) -> float:
    """Convert target grade level to steering strength using regression model."""
    return (target_grade - REGRESSION_INTERCEPT) / REGRESSION_SLOPE


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    complexity_vector: torch.Tensor,
    alpha: float,
) -> str:
    """Generate text with specified steering strength."""
    # Format prompt using chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Apply steering
    steering_vector = complexity_vector.to(model.device) * alpha

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
                do_sample=False,  # Greedy decoding for determinism
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        hook_handle.remove()

    # Extract generated portion (after chat template)
    if "<|im_start|>assistant\n" in full_text:
        generated_text = full_text.split("<|im_start|>assistant\n", 1)[1]
    elif "assistant\n" in full_text:
        generated_text = full_text.split("assistant\n", 1)[1]
    else:
        generated_text = full_text

    # Clean up special tokens
    for token in ["<|im_end|>", "<|endoftext|>"]:
        if token in generated_text:
            generated_text = generated_text.split(token)[0]

    return generated_text.strip()


def analyze_text(text: str) -> float:
    """Compute Flesch-Kincaid grade level."""
    if not text or len(text.strip()) < 10:
        return 0.0

    try:
        return textstat.flesch_kincaid_grade(text)
    except:
        return 0.0


def run_calibration_test(model, tokenizer, complexity_vector):
    """Run calibration test across target grade levels."""
    print(f"\n{'='*80}")
    print(f"Running Calibration Test")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  - Prompt: {TEST_PROMPT}")
    print(f"  - Target grades: {TARGET_GRADES[0]} to {TARGET_GRADES[-1]}")
    print(f"  - Max tokens: {MAX_NEW_TOKENS}")
    print(f"\n")

    results = []

    for target_grade in TARGET_GRADES:
        alpha = grade_to_alpha(target_grade)

        print(f"Target grade {target_grade:2d} (α = {alpha:+.3f})...", end=" ", flush=True)

        # Generate text
        generated_text = generate_with_steering(
            model, tokenizer, TEST_PROMPT, complexity_vector, alpha
        )

        # Measure actual grade level
        actual_grade = analyze_text(generated_text)

        # Store results
        results.append({
            'target_grade': target_grade,
            'alpha': alpha,
            'actual_grade': actual_grade,
            'error': actual_grade - target_grade,
            'generated_text': generated_text,
        })

        print(f"Actual: {actual_grade:.1f} (error: {actual_grade - target_grade:+.1f})")

    return pd.DataFrame(results)


def visualize_calibration(df: pd.DataFrame):
    """Create calibration plot."""
    print(f"\n{'='*80}")
    print(f"Creating Calibration Plot")
    print(f"{'='*80}\n")

    # Split data into effective range and outliers
    df_effective = df[df['target_grade'] <= 16].copy()
    df_outliers = df[df['target_grade'] > 16].copy()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot of actual vs target (effective range only)
    ax.scatter(df_effective['target_grade'], df_effective['actual_grade'],
               s=100, alpha=0.7, color='#2E86AB', edgecolors='white', linewidth=2,
               label='Measured grade level', zorder=3)

    # Perfect calibration line (y = x)
    min_grade = df_effective['target_grade'].min()
    max_grade = max(df_effective['target_grade'].max(), df_effective['actual_grade'].max())
    ax.plot([min_grade, max_grade], [min_grade, max_grade],
            'k--', linewidth=2, alpha=0.5, label='Perfect calibration (y = x)', zorder=1)

    # Best fit line (using effective range only)
    z = np.polyfit(df_effective['target_grade'], df_effective['actual_grade'], 1)
    p = np.poly1d(z)
    x_line = np.array([df_effective['target_grade'].min(), df_effective['target_grade'].max()])
    ax.plot(x_line, p(x_line),
            color='#A23B72', linewidth=2, linestyle='-',
            label=f'Best fit: y = {z[0]:.2f}x + {z[1]:.2f}', zorder=2)

    # Add arrow and annotation for outliers
    if len(df_outliers) > 0:
        # Draw arrow from grade 16 point pointing upward/right
        # last_point = df_effective[df_effective['target_grade'] == 16].iloc[0]

        # Arrow starting from the last good point
        # arrow_start_x = last_point['target_grade']
        # arrow_start_y = last_point['actual_grade']

        # ax.annotate('',
        #            xy=(17.5, max_grade * 0.95),
        #            xytext=(arrow_start_x, arrow_start_y),
        #            arrowprops=dict(arrowstyle='->', color='#E63946', lw=2.5, linestyle='--'),
        #            zorder=2)

        # Add text annotations for the outlier values
        outlier_text = "Beyond effective range:\n"
        for _, row in df_outliers.iterrows():
            outlier_text += f"  Grade {int(row['target_grade'])}: FK={row['actual_grade']:.0f}\n"

        ax.text(0.98, 0.75, outlier_text.strip(),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#FFE5E5', edgecolor='#E63946', alpha=0.9),
                color='#E63946',
                fontweight='bold')

    # Labels and styling
    ax.set_xlabel('Target Grade Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual Grade Level', fontsize=13, fontweight='bold')
    ax.set_title('Grade Level Calibration Test\n(How well does our model predict output complexity?)',
                 fontsize=15, fontweight='bold', pad=20)

    # Equal aspect ratio for fair comparison
    ax.set_aspect('equal', adjustable='box')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # Compute calibration metrics (using effective range only)
    mae = np.mean(np.abs(df_effective['error']))
    rmse = np.sqrt(np.mean(df_effective['error']**2))
    r_squared = np.corrcoef(df_effective['target_grade'], df_effective['actual_grade'])[0, 1]**2

    # Add stats box
    stats_text = f'Effective range (grades 6-16):\nMAE = {mae:.2f} grades\nRMSE = {rmse:.2f} grades\nR² = {r_squared:.3f}'
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    plot_path = OUTPUT_DIR / "grade_level_calibration.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved calibration plot to {plot_path}")

    plt.close()

    return mae, rmse, r_squared


def main():
    """Run grade level calibration test."""
    print("\n" + "="*80)
    print("GRADE LEVEL CALIBRATION TEST")
    print("="*80)

    # Load model and vectors
    model, tokenizer = load_model_and_tokenizer()
    complexity_vector = load_complexity_vector()

    # Run calibration test
    df = run_calibration_test(model, tokenizer, complexity_vector)

    # Save results
    csv_path = OUTPUT_DIR / "calibration_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")

    # Create visualization
    mae, rmse, r_squared = visualize_calibration(df)

    # Print summary
    print(f"\n{'='*80}")
    print(f"CALIBRATION SUMMARY")
    print(f"{'='*80}\n")
    print(f"Mean Absolute Error (MAE): {mae:.2f} grades")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} grades")
    print(f"R² correlation: {r_squared:.3f}")
    print(f"\nWorst predictions:")
    worst = df.nlargest(3, 'error', keep='all')[['target_grade', 'actual_grade', 'error']]
    print(worst.to_string(index=False))

    print("\n" + "="*80)
    print("CALIBRATION TEST COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Plot: {OUTPUT_DIR / 'grade_level_calibration.png'}")
    print("\n")


if __name__ == "__main__":
    main()
