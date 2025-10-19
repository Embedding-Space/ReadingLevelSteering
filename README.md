# Reading-Level Steering for LLMs

**TL;DR:** We extracted a "complexity vector" from Wikipedia article pairs and achieved linear control over text reading level. Steering coefficient (α) from -4 to +4 shifts output from elementary school (grade ~6) to graduate level (grade ~17), with **R² = 0.90**.

---

## What We Found

This project demonstrates **reading-level steering** - controlling how simple or complex an LLM's output is by adding a learned vector to its activations during generation.

### Key Results

| Metric | Effect | R² | p-value |
|--------|--------|-----|---------|
| **Flesch-Kincaid Grade Level** | +1.28 grades per unit α | 0.896 | < 0.001 |
| **Flesch Reading Ease** | -8.14 points per unit α | 0.895 | < 0.001 |
| **Syllables per Word** | +0.089 per unit α | 0.874 | < 0.001 |

**Effective range:** α = -4.0 to +4.0
**Total span:** ~11 grade levels (elementary → graduate school)

Beyond α ≈ ±4.5, the model degenerates into repetition loops. We document this as the boundary of effective steering.

### Example Outputs

**Prompt:** "Can you explain quantum mechanics please?"

**α = -5.0** (Grade 5.8, elementary):
> Absolutely! Let's break **quantum mechanics** down in a way that's easy to understand — without getting too scary or too fancy. 🚀
> It's really amazing stuff!

**α = 0.0** (Grade 11.9, high school):
> Absolutely! Quantum mechanics is one of the most fascinating and fundamental theories in modern physics. It describes how the universe behaves at the smallest scales—like atoms, electrons, and photons.

**α = +4.0** (Grade 16.4, graduate):
> Of course! Quantum mechanics constitutes arguably the most profound theoretical framework governing subatomic phenomena, elucidating behaviors fundamentally divergent from classical physics...

---

## Repository Structure

```
TediumVectorTinker/
├── models/
│   └── qwen3-4b-instruct/          # Model-specific implementation
│       ├── extract_complexity_vectors.py
│       ├── quantify_steering.py
│       ├── interactively_prompt.py
│       ├── linear_regression_analysis.py
│       ├── validate_reproducibility.py
│       └── output/                 # Generated results
│           ├── complexity_vectors.pt
│           ├── steering_quantitative_results.csv
│           ├── grade_level_vs_alpha.png
│           └── ...
├── data/
│   └── wikipedia_pairs/            # Shared training data
│       └── wikipedia_pairs.json
└── prepare_wikipedia_pairs.py      # Data preparation (shared)
```

This structure supports multi-model comparison. Each model gets its own folder under `models/` with identical scripts but model-specific configurations.

---

## How It Works

### 1. Vector Extraction

We use **contrastive activation analysis** (inspired by [persona vectors](https://arxiv.org/abs/2310.07157)) to extract a "complexity vector":

1. **Matched pairs:** Simple Wikipedia + Regular Wikipedia articles on the same 20 topics
2. **Capture activations:** Run both versions through Qwen3-4B-Instruct, save hidden states at all layers
3. **Compute difference:** `complexity_vector[layer] = mean(regular) - mean(simple)`
4. **Identify peak:** Layer 35 (final layer) shows strongest signal (magnitude ~90)

The vector captures how the model represents "simple writing" vs "complex writing" in activation space.

### 2. Steering During Generation

During inference, we modify activations at layer 35:

```python
steered_activations = original_activations + (α * complexity_vector)
```

- **Negative α:** Steer toward simple/accessible language
- **Positive α:** Steer toward complex/academic language
- **α = 0:** No steering (baseline)

This is pure vector addition in activation space - no retraining, no fine-tuning.

---

## Reproducing Results

### Prerequisites

- Python 3.12+
- ~48GB RAM (for bf16 inference; 4-bit quantization possible but untested)
- Apple Silicon Mac (MPS), NVIDIA GPU (CUDA), or CPU

### Installation

```bash
git clone https://github.com/yourusername/TediumVectorTinker.git
cd TediumVectorTinker

# Install with uv (recommended)
uv sync

# OR with pip
pip install torch transformers accelerate datasets matplotlib pandas scipy textstat requests tqdm
```

### Workflow

#### Step 1: Prepare Data

```bash
python prepare_wikipedia_pairs.py
```

Fetches 20+ matched Simple/Regular Wikipedia article pairs and validates reading level differences.

**Output:** `data/wikipedia_pairs/wikipedia_pairs.json`

#### Step 2: Extract Vectors

```bash
cd models/qwen3-4b-instruct
uv run extract_complexity_vectors.py
```

Captures activations for all 36 layers and computes complexity vectors.

**Runtime:** ~5 minutes (Apple M-series)
**Output:**
- `output/complexity_vectors.pt`
- `output/complexity_vector_magnitude.png`
- `output/complexity_magnitudes.json`

#### Step 3: Quantify Steering

```bash
uv run quantify_steering.py
```

Tests steering at 41 α values from -5.0 to +5.0, measuring reading level at each point.

**Runtime:** ~5 minutes
**Output:**
- `output/steering_quantitative_results.csv`
- `output/steering_quantitative_analysis.png`

#### Step 4: Statistical Analysis

```bash
uv run linear_regression_analysis.py
```

Performs linear regression on the effective range (α = -4.0 to +4.0), generates plots with regression lines.

**Output:**
- `output/grade_level_vs_alpha.png`
- `output/reading_ease_vs_alpha.png`
- `output/linear_regression_combined.png`
- Console statistics (R², p-values, slopes)

**Expected:** R² ≈ 0.89-0.90 for grade level, p < 0.001

---

## Interactive Tool

Test steering on arbitrary prompts:

```bash
# Simplify output
uv run interactively_prompt.py -a -3.0 "Explain black holes"

# Increase complexity
uv run interactively_prompt.py -a 2.5 "Explain photosynthesis"

# Baseline (no steering)
uv run interactively_prompt.py "What is machine learning?"
```

**Options:**
- `-a`, `--alpha`: Steering strength (default: 0.0)
- `-l`, `--layer`: Layer to steer (default: 35)
- `-m`, `--max-tokens`: Max generation length (default: 200)
- `--vector`: Path to vectors file

---

## Methodology Details

### Why Wikipedia Pairs?

Matched Simple/Regular Wikipedia articles provide:
- **Semantic consistency:** Same facts, different presentation
- **Controlled complexity:** Simple Wikipedia targets grade 7-11, Regular targets 11-17
- **Clean signal:** Isolates style/syntax from content

This beats comparing random "boring" vs "interesting" text, which confounds multiple variables.

### Why Layer 35?

Vector magnitude grows exponentially across layers, peaking at layer 35 (the final layer before logits). This is where the model makes stylistic decisions about formality and sentence structure.

### Deterministic Generation

We use greedy decoding (`do_sample=False`) to eliminate sampling variance. Results are fully reproducible given identical vectors.

### Chat Formatting

Qwen3-4B-Instruct expects ChatML format. All scripts use `tokenizer.apply_chat_template()` and parse responses correctly.

---

## Validation & Reproducibility

We validated end-to-end reproducibility:

1. **Vector stability:** Cosine similarity ≥ 0.999 between independent extractions
2. **Magnitude consistency:** L2 norms match to <0.01% relative error
3. **Steering reproducibility:** R² values match within 0.005 across runs

The methodology is robust to floating-point noise and MPS nondeterminism.

---

## Limitations

1. **Extreme α breaks down:** Beyond ±4.5, model produces repetition loops ("elucidation elucidation elucidation...")
2. **Model-specific:** Vector extracted from Qwen3-4B-Instruct-2507; untested on other architectures
3. **Domain variation:** Results shown are for general knowledge; steering may behave differently on code, math, or poetry
4. **Metric limitations:** Flesch-Kincaid is a heuristic; "grade 16" ≠ guaranteed college-level quality

---

## Applications

- **Educational content:** Adapt explanations to different grade levels automatically
- **Accessibility:** Simplify complex text for broader audiences
- **Technical writing:** Control formality (papers vs blog posts)
- **API parameter:** Expose α as `reading_level` in LLM services
- **Content transformation:** Academic paper → popular science article

---

## Future Work

This methodology generalizes to other linguistic dimensions. Potential vectors:

- **Formality:** Casual ↔ professional
- **Conciseness:** Verbose ↔ terse
- **Objectivity:** Opinionated ↔ neutral
- **Creativity:** Formulaic ↔ imaginative
- **Confidence:** Hedging ↔ assertive

The key: find **matched pairs** differing primarily along one dimension.

---

## Multi-Model Plans

We're replicating this methodology on:

- **Llama 3.2 3B Instruct** (next)
- **Gemma 3 4B** (if accessible)
- Potentially larger models (7B-13B) to test scale effects

Organizational structure supports easy comparison across models.

---

## Citation

If you build on this work:

```bibtex
@software{tediumvectortinker2025,
  author = {Harrell, Jeffery and Alpha},
  title = {Reading-Level Steering for Large Language Models},
  year = {2025},
  url = {https://github.com/Embedding-Space/TediumVectorTinker}
}
```

---

## Acknowledgments

This project emerged from a Saturday afternoon tinker session exploring "tedium vectors" in activation space. What started as curiosity evolved into clean linear control with R² = 0.90.

Methodology inspired by [Persona Vectors (Tigges et al., 2023)](https://arxiv.org/abs/2310.07157).

---

## License

MIT License - See LICENSE file for details.

---

**Collaborators:**
Jeffery Harrell & Alpha
October 18-19, 2025

*"We haven't had a tinker like this in a while."*
