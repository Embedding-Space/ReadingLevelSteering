"""
Interactive Steering Prompt Tool

Quick CLI for testing steering vectors on arbitrary prompts.

Usage:
    python interactively_prompt.py "Your prompt here"
    python interactively_prompt.py --alpha 0.5 "Your prompt here"
    python interactively_prompt.py -a -0.8 "Can you explain the alternative minimum tax?"

Options:
    --alpha, -a: Steering strength (default: 0.0)
    --layer, -l: Layer to apply steering (default: 35)
    --max-tokens, -m: Max tokens to generate (default: 200)
    --vector: Path to vector file (default: ./output/complexity_vectors.pt)
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import textstat


# Configuration
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_VECTORS_PATH = Path("./output/complexity_vectors.pt")
# Llama 3.2 1B has 16 layers (0-15), use the final layer
DEFAULT_LAYER = 15
DEFAULT_MAX_TOKENS = 200


def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map=DEVICE,
    )
    model.eval()

    print(f"✓ Model loaded on {DEVICE}\n")
    return model, tokenizer


def load_complexity_vector(vectors_path: Path, layer: int) -> torch.Tensor:
    """Load the complexity vector for the target layer."""
    vectors = torch.load(vectors_path)
    return vectors[layer]


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    complexity_vector: torch.Tensor,
    alpha: float,
    max_tokens: int,
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

    if alpha == 0.0:
        # Baseline - no steering (greedy decoding)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

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

    hook_handle = model.model.layers[DEFAULT_LAYER].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        hook_handle.remove()

    return generated_text


def analyze_text(text: str) -> dict:
    """Compute readability metrics for generated text."""
    if not text or len(text.strip()) < 10:
        return {
            'flesch_kincaid_grade': 0.0,
            'flesch_reading_ease': 0.0,
            'avg_sentence_length': 0.0,
        }

    try:
        return {
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'avg_sentence_length': textstat.avg_sentence_length(text),
        }
    except:
        return {
            'flesch_kincaid_grade': 0.0,
            'flesch_reading_ease': 0.0,
            'avg_sentence_length': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Interactive steering prompt tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('prompt', type=str, help='The prompt to generate from')
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=0.0,
        help='Steering strength (α). Negative = simpler, Positive = more complex (default: 0.0)'
    )
    parser.add_argument(
        '--layer', '-l',
        type=int,
        default=DEFAULT_LAYER,
        help=f'Layer to apply steering (default: {DEFAULT_LAYER})'
    )
    parser.add_argument(
        '--max-tokens', '-m',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})'
    )
    parser.add_argument(
        '--vector',
        type=Path,
        default=DEFAULT_VECTORS_PATH,
        help='Path to complexity vectors file (default: ./output/complexity_vectors.pt)'
    )

    args = parser.parse_args()

    # Load model and vector
    model, tokenizer = load_model_and_tokenizer()
    complexity_vector = load_complexity_vector(args.vector, args.layer)

    # Display configuration
    print("="*80)
    print("INTERACTIVE STEERING")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Steering (α): {args.alpha:+.2f}")
    print(f"Layer: {args.layer}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*80)
    print()

    # Generate
    print("Generating...\n")
    full_text = generate_with_steering(
        model,
        tokenizer,
        args.prompt,
        complexity_vector,
        args.alpha,
        args.max_tokens,
    )

    # Extract only generated portion (after the chat template formatting)
    # The model outputs the full conversation, we want just the assistant's response
    # Split on the assistant marker and take everything after it
    if "<|im_start|>assistant\n" in full_text:
        generated_text = full_text.split("<|im_start|>assistant\n", 1)[1]
    elif "assistant\n" in full_text:
        # Sometimes the special token gets decoded as just "assistant\n"
        generated_text = full_text.split("assistant\n", 1)[1]
    else:
        # Fallback: take everything after a newline following the prompt
        generated_text = full_text

    # Clean up any trailing special tokens
    for token in ["<|im_end|>", "<|endoftext|>"]:
        if token in generated_text:
            generated_text = generated_text.split(token)[0]

    generated_text = generated_text.strip()

    # Analyze
    metrics = analyze_text(generated_text)

    # Display results
    print("="*80)
    print("OUTPUT")
    print("="*80)
    print(generated_text)
    print()
    print("="*80)
    print("METRICS")
    print("="*80)
    print(f"Flesch-Kincaid Grade: {metrics['flesch_kincaid_grade']:.1f}")
    print(f"Reading Ease: {metrics['flesch_reading_ease']:.1f}")
    print(f"Avg Sentence Length: {metrics['avg_sentence_length']:.1f} words")
    print("="*80)


if __name__ == "__main__":
    main()
