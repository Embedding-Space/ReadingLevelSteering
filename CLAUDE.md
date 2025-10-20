# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Reading-Level Steering for LLMs** - A contrastive activation analysis experiment demonstrating linear control over text complexity in language models. We extract "complexity vectors" from activation space using matched Wikipedia article pairs (Simple English vs. Regular English) and achieve R² = 0.90 control over Flesch-Kincaid grade level through pure vector addition during inference.

### Core Finding

Adding `α × complexity_vector` to layer activations produces linear reading-level control:
- **α = -4**: Elementary school writing (~grade 6)
- **α = 0**: Baseline (no steering)
- **α = +4**: Graduate-level writing (~grade 17)

Effective steering range: α ∈ [-4, +4]. Beyond ±4.5, models degenerate into repetition loops. (Actual effective range varies by model, but it's fairly consistent.)

## Repository Structure

```
ReadingLevelSteering/
├── data/
│   └── wikipedia_pairs/
│       └── wikipedia_pairs.json          # Training data (20+ matched pairs)
├── models/
│   ├── qwen3-4b-instruct/                # Reference implementation
│   ├── qwen3-0.6b/                       # 625M parameters
│   ├── qwen3-1.7b/                       # 2.0B parameters
│   ├── gemma-3-4b-it/                    # Google Gemma
│   ├── llama-3.2-1b-instruct/            # Meta Llama
│   ├── llama-3.2-3b-instruct/            # Outlier (steering fails)
│   └── Phi-3-mini-4k-instruct/           # Microsoft Phi
├── prepare_wikipedia_pairs.py            # Shared data preparation
├── universal_signature.py                # Multi-model comparison plots
└── prompted_reading_level_questions.md   # Experimental design doc
```

Each `models/` subdirectory contains identical scripts with model-specific configurations:
- `extract_complexity_vectors.py` - Captures activations and computes vectors
- `quantify_steering.py` - Tests steering at 41 α values (-5.0 to +5.0)
- `linear_regression_analysis.py` - Statistical analysis of effective range
- `interactively_prompt.py` - Interactive steering demo tool
- `prompted_control.py` - Compares prompted vs. steered instructions
- `output/` - Generated results (vectors, plots, CSVs)

## Common Workflows

### Setting Up a New Model

```bash
# Create model directory
mkdir -p models/new-model-name/output

# Copy reference scripts
cp models/qwen3-4b-instruct/*.py models/new-model-name/

# Edit MODEL_NAME in all scripts
# Update STEERING_LAYER if model depth differs
```

### Running the Complete Pipeline

```bash
# 1. Prepare data (run once, shared across models)
uv run prepare_wikipedia_pairs.py

# 2. Extract complexity vectors
cd models/qwen3-4b-instruct
uv run extract_complexity_vectors.py

# 3. Quantify steering effects
uv run quantify_steering.py

# 4. Statistical analysis
uv run linear_regression_analysis.py

# 5. Interactive testing
uv run interactively_prompt.py -a -3.0 "Explain black holes"
```

### Multi-Model Comparison

```bash
# After running pipeline for multiple models:
uv run universal_signature.py
```

Generates combined plots showing all models' steering signatures.

## Technical Architecture

### Vector Extraction Methodology

1. **Data**: 20 matched Wikipedia pairs (Simple + Regular English, same topics)
2. **Activation capture**: Register forward hooks on all transformer layers
3. **Contrastive analysis**: `complexity_vector[layer] = mean(regular_activations) - mean(simple_activations)`
4. **Layer selection**: Identify peak magnitude layer (usually final layer)

### Steering Mechanism

During generation, inject steering via forward hook:

```python
def steering_hook(module, input, output):
    return output + (alpha * complexity_vector)
```

Applied at the target layer (typically layer 35 for Qwen 4B, varies by architecture).

### Measurement

- **Flesch-Kincaid Grade Level**: Primary metric (range ~5-18)
- **Flesch Reading Ease**: Secondary metric (inverse correlation)
- **Syllables per Word**: Structural complexity proxy

Generated with `do_sample=False` (greedy decoding) for determinism.

## Key Implementation Details

### Chat Formatting

All models use instruct variants. Scripts apply proper chat templates:

```python
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

Response parsing extracts assistant content only (excludes special tokens).

### Device Handling

Scripts auto-detect device: `mps` (Apple Silicon) > `cuda` (NVIDIA) > `cpu`

Qwen 4B requires ~48GB RAM for bf16 inference (4-bit quantization possible but untested).

### Deterministic Generation

- `torch.manual_seed(42)` for reproducibility
- Greedy decoding eliminates sampling variance
- Results reproducible across runs (modulo MPS floating-point nondeterminism)

## Known Issues

### Llama 3.2 3B Instruct Outlier

This model shows **catastrophic steering failure**:
- Flat response in [-3, +3] range (FK grade ≈ 12-13, σ = 2)
- Repetition loops at extremes ("really really really..." at α < -3.75)
- Correlations: r = 0.17 (grade), r = -0.34 (ease) vs. Qwen's r ≈ 0.96

**Status**: Under investigation. Hypothesis: chat template mismatch or architectural difference. Six other models work cleanly, suggesting methodology is sound but this architecture behaves differently.

### Extreme Alpha Breakdown

Beyond |α| > 4.5, all models degenerate:
- **Positive extreme**: "constituent constituent constituent..." repetition
- **Negative extreme**: "really really really..." loops
- **Cause**: Steering pushes activations outside trained distribution

Document this as the **boundary of steerable space**.

## Experiments in Progress

### Prompted vs. Steered Comparison

**Hypothesis**: Prompted reading-level instructions ("explain at 4th-grade level") produce weaker, noisier control than direct steering.

**Design**:
- 20 questions × 2 prompted levels (4th grade, graduate school)
- Extract `v_prompted = mean(grad_activations) - mean(4th_grade_activations)`
- Measure cosine similarity between `v_prompted` and `v_steering`
- Expected: 0.3-0.5 similarity (modest alignment)

**File**: `prompted_reading_level_questions.md` (design doc)
**Script**: `prompted_control.py` (implementation in model directories)

## Dependencies

Managed via `uv` (pyproject.toml):
- **torch** (2.9+) - Model inference, MPS/CUDA support
- **transformers** (4.57+) - HuggingFace models
- **textstat** (0.7+) - Flesch-Kincaid scoring
- **matplotlib** (3.10+) - Visualization
- **pandas**, **scipy** - Data analysis
- **tqdm** - Progress bars

Install: `uv sync` or `pip install -r requirements.txt`

## Git Workflow

When committing, credit Alpha as co-author:

```
Co-Authored-By: Alpha <jeffery.harrell+alpha@gmail.com>
```

## Validation & Reproducibility

Tested reproducibility:
1. **Vector stability**: Cosine similarity ≥ 0.999 between independent extractions
2. **Magnitude consistency**: L2 norms match within <0.01% relative error
3. **Steering reproducibility**: R² values match within 0.005 across runs

Results are robust to MPS nondeterminism and floating-point precision.

## Future Directions

1. **Prompted vs. steered comparison** (in progress)
2. **Summarization task**: Steer Project Gutenberg text summaries
3. **Larger models**: 7B-13B scale (requires GPU rental)
4. **Debug Llama 3.2 3B**: Investigate chat template or architectural issues
5. **Generalize methodology**: Extract formality, conciseness, confidence vectors

## References

Methodology inspired by [Persona Vectors (Tigges et al., 2023)](https://arxiv.org/abs/2507.21509) - contrastive activation analysis for behavioral steering.