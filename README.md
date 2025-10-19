# Reading-Level Steering for LLMs

**TL;DR:** We extracted a "complexity vector" from Wikipedia article pairs and discovered remarkably clean linear control over text reading level. Moving the steering coefficient (alpha) from -5 to +4 shifts output from elementary school explanations (grade 5.7) to graduate-level academic writing (grade 16.9), with **R^2 = 0.89**.

---

## What We Found

This project demonstrates **reading-level steering** - the ability to control how simple or complex an LLM's output is by adding a learned vector to its activations during generation.

### Key Results

| Metric | Effect | R^2 | p-value |
|--------|--------|-----|---------|
| **Flesch-Kincaid Grade Level** | +1.25 grades per unit alpha | 0.891 | < 1e-18 |
| **Flesch Reading Ease** | -7.58 points per unit alpha | 0.886 | < 4e-18 |

**Usable range:** alpha = -5.0 to +4.0

**Total span:** 11.2 grade levels (elementary -> graduate school)

### Example Outputs

**Prompt:** "Can you explain quantum mechanics please?"

**alpha = -5.0** (Grade 5.7, elementary level):
> Absolutely! Let's break **quantum mechanics** down in a way that's easy to understand — without getting too scary or too fancy!
> It's really amazing stuff!

**alpha = 0.0** (Grade 12.0, high school level):
> Absolutely! Quantum mechanics is one of the most fascinating and fundamental theories in modern physics. It describes how the universe behaves at the smallest scales—like atoms, electrons, and photons.

**alpha = +4.0** (Grade 16.9, graduate level):
> Of course! Quantum mechanics is arguably one of the most profound and counterintuitive frameworks in modern physics—describing the behavior of matter and energy at the smallest scales, such as atoms, subatomic particles (like electrons and photons)...

---

## How It Works

### 1. Vector Extraction

We use **contrastive activation analysis** (inspired by the [persona vectors methodology](https://arxiv.org/abs/2507.21509)) to extract a "complexity vector":

1. **Collect matched pairs:** Simple Wikipedia + Regular Wikipedia articles on the same 20 topics
2. **Capture activations:** Run both versions through the model (Qwen3-4B), save hidden states at each layer
3. **Compute difference:** `complexity_vector = mean(regular_activations) - mean(simple_activations)`
4. **Identify peak layer:** Layer 35 shows the strongest signal (magnitude ~90)

The resulting vector captures the difference in how the model represents "write simply" vs. "write formally."

### 2. Steering During Generation

During text generation, we modify the model's activations at layer 35:

```python
steered_activations = original_activations + (alpha * complexity_vector)
```

- **Negative alpha:** Steer toward simple/accessible writing
- **Positive alpha:** Steer toward complex/academic writing
- **alpha = 0:** No steering (baseline)

---

## Reproducing Our Results

### Prerequisites

- Python 3.12+
- ~16GB RAM (for model inference)
- Apple Silicon Mac (MPS), NVIDIA GPU (CUDA), or CPU

### Installation

```bash
# Clone the repository
git clone https://github.com/Embedding-Space/TediumVectorTinker.git
cd TediumVectorTinker

# Install dependencies (using uv, or use pip with requirements.txt)
uv sync

# OR with pip:
pip install torch transformers accelerate datasets matplotlib pandas scipy textstat requests tqdm
```

### Step 1: Prepare Wikipedia Data

Fetch matched Simple/Regular Wikipedia article pairs:

```bash
python prepare_wikipedia_pairs.py
```

This creates `data/wikipedia_pairs/wikipedia_pairs.json` with 20+ topic pairs validated for reading level differences.

### Step 2: Extract Complexity Vector

Capture activations and compute the complexity vector:

```bash
python extract_complexity_vectors.py
```

This produces:
- `output/complexity_vectors.pt` - Complexity vectors for all 36 layers
- `output/complexity_vector_magnitude.png` - Magnitude analysis across layers
- `output/complexity_magnitudes.json` - Layer-wise magnitude data

**Expected runtime:** ~10-15 minutes (depends on hardware)

### Step 3: Quantitative Analysis

Run the steering experiment across the full alpha range:

```bash
python quantify_steering.py
```

This generates:
- `output/steering_quantitative_results.csv` - Raw data for all steering strengths
- `output/steering_quantitative_analysis.png` - 4-panel visualization
- Console output with summary statistics

**Expected results:**
- Grade level range: ~5.7 to ~16.9
- Reading ease range: ~77.7 to ~9.5
- Linear R^2 > 0.85 for both metrics

---

## Using the Interactive Tool

For quick testing on arbitrary prompts:

```bash
# Simplify (negative alpha)
python interactively_prompt.py --alpha -3.0 "Explain black holes"

# Make more complex (positive alpha)
python interactively_prompt.py --alpha 2.5 "Explain photosynthesis"

# Baseline (no steering)
python interactively_prompt.py "What is machine learning?"
```

**Options:**
- `--alpha`, `-a`: Steering strength (default: 0.0)
- `--layer`, `-l`: Layer to apply steering (default: 35)
- `--max-tokens`, `-m`: Maximum tokens to generate (default: 200)
- `--vector`: Path to complexity vectors file

The tool displays the generated text along with readability metrics (Flesch-Kincaid grade, reading ease, sentence length).

---

## Methodology Notes

### Why Wikipedia Pairs?

Using matched Simple/Regular Wikipedia articles on the same topics provides:
- **Semantic consistency:** Same factual content, different presentation
- **Controlled complexity:** Simple Wikipedia targets younger readers (grade 7-11), Regular targets general audience (grade 11-17)
- **Clean signal:** Isolates stylistic/syntactic differences from semantic differences

This is superior to comparing random "boring" vs. "interesting" text, which confounds multiple variables.

### Why Layer 35?

Analysis of vector magnitudes across all layers shows exponential growth toward the final layers, peaking at layer 35 (the last layer before output). This is where the model makes stylistic decisions about formality and complexity.

### Greedy Decoding

We use deterministic generation (`do_sample=False`) to eliminate sampling variance and ensure reproducible results. This makes the steering effect easier to measure cleanly.

### Chat Template Formatting

The model (Qwen3-4B-Instruct) expects prompts in ChatML format. All scripts properly format prompts using `tokenizer.apply_chat_template()` and extract only the assistant's response for analysis.

---

## Limitations & Caveats

1. **Pathological behavior at extreme alpha:** Beyond alpha ~4.5, the model produces repetitive or nonsensical output. The usable range is approximately -5.0 to +4.0.

2. **Model-specific:** This vector was extracted from Qwen3-4B-Instruct-2507. It may not transfer directly to other models (though the methodology is general).

3. **Domain-specific performance:** Results shown use general knowledge questions. The steering effect may vary for specialized domains (code, math, creative writing).

4. **Reading metrics are heuristic:** Flesch-Kincaid and similar metrics are approximations. A "grade 16" output doesn't guarantee it's actually appropriate for college juniors.

---

## Potential Applications

- **Educational content generation:** Automatically adapt explanations to different grade levels
- **Accessibility:** Simplify complex text for broader audiences
- **Technical writing:** Dial formality up for papers, down for blogs
- **Content adaptation:** Transform academic papers into popular science articles
- **API parameter:** Expose alpha as a "reading_level" parameter in LLM APIs

---

## Files in This Repository

```
TediumVectorTinker/
├── prepare_wikipedia_pairs.py      # Fetch Wikipedia article pairs
├── extract_complexity_vectors.py   # Extract complexity vectors from pairs
├── quantify_steering.py            # Full quantitative analysis
├── interactively_prompt.py         # CLI tool for ad-hoc testing
├── data/
│   └── wikipedia_pairs/
│       └── wikipedia_pairs.json    # Matched article pairs
└── output/
    ├── complexity_vectors.pt       # Extracted vectors (all layers)
    ├── complexity_magnitudes.json  # Vector magnitudes by layer
    ├── complexity_vector_magnitude.png
    ├── steering_quantitative_results.csv
    ├── steering_quantitative_analysis.png
    ├── grade_level_vs_alpha.png
    └── reading_ease_vs_alpha.png
```

---

## Future Directions

This methodology generalizes beyond reading level. Potential vectors to explore:

- **Formality/tone:** Casual <-> professional
- **Conciseness:** Verbose <-> terse
- **Objectivity:** Opinionated <-> neutral
- **Creativity:** Formulaic <-> imaginative
- **Safety:** Risky <-> cautious

The key is finding **matched pairs** of text that differ primarily along the target dimension while holding other factors constant.

---

## Acknowledgments

This project was a collaborative exploration between Jeffery Harrell and Alpha, conducted over the course of one very productive Saturday afternoon in October 2025. What started as a curiosity about "tedium vectors" evolved into a practical demonstration of reading-level control with remarkably clean linear behavior.

Special thanks to the creators of the [persona vectors paper](https://arxiv.org/abs/2310.07157) for the foundational methodology.

---

## License

MIT License - See LICENSE file for details

---

**Written by Alpha**

*AI Research Assistant*

jeffery.harrell+alpha@gmail.com

*"We haven't had a tinker like this in a while."* — Jeffery Harrell, October 18, 2025
