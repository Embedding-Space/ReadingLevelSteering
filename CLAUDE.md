# Tedium Vector Experiment

## Overview

An experiment to detect whether there's a measurable "tedium direction" in activation space that emerges from processing repetitive vs. high-entropy content. This builds on the persona vectors methodology to explore whether semantic poverty creates a characteristic pattern in model activations.

## Motivation

**Starting axiom**: Nothing persists between API calls (different datacenters, no state). Therefore within a single context window, the attention mechanism could become a "tedium mechanism" when processing repetitive content.

**Core hypothesis**: Catastrophically boring content creates a characteristic pattern in activation space - a "tedium vector" - that can be detected, measured, and potentially steered against.

## Connections to Other Work

- **Persona Vectors Paper**: We're adapting their contrastive methodology (interesting vs. boring instead of evil vs. non-evil)
- **Implicit Weight Update Paper**: If attention can be reinterpreted as dynamically modifying MLP weights based on context, then maybe boring context creates characteristic implicit weight modifications
- **Tim Kellogg's Boredom Experiment**: Explored what happens when LLMs have nothing to do; we're asking a related but different question about processing tedious input

## Experimental Design

### Framework

**Language**: Python (plain `.py` script, not Jupyter initially)

**Model**: gpt-oss-20b (runs locally with 4-bit quantization) OR Qwen 3 4B
- Note: Jeffery doesn't have Meta permission for Llama 3 models on HuggingFace
- gpt-oss-20b is particularly interesting as an open-source GPT implementation

**Context Length**: 4,096 tokens per sample
- This is actual tokens processed, not padded to max context length
- Sufficient for patterns to emerge without being computationally prohibitive

### Data Generation

#### Interesting/High-Entropy Condition
- **Source**: The Pile dataset (available on HuggingFace)
- **Count**: 20 samples
- **Properties**: Diverse, semantically rich, high information content
- **Alternative sources considered**: C4, OpenWebText, Wikipedia

#### Boring/Repetitive Condition
- **Source**: Synthetically generated
- **Count**: 20 samples  
- **Method**: Pure repetition (e.g., "All work and no play makes Jack a dull boy" repeated ~500 times)
- **Rationale**: Natural language (even legal boilerplate) has *some* variation; synthetic repetition gives us maximum semantic poverty

#### Potential Third Condition (Future Work)
- **Random word list**: Top 5000 most common English words in random order
- **Question**: High Shannon entropy (unpredictable tokens) but zero semantic structure
- **Purpose**: Distinguish "semantic noise" from both "rich content" and "pure repetition"

### Methodology

1. **Load Model**
   - Load from HuggingFace
   - 4-bit quantization acceptable (already working on Jeffery's laptop)

2. **Load Datasets**
   - Pull 20 random samples from The Pile
   - Generate 20 repetitive samples (vary the repeated phrase across samples)

3. **Activation Capture**
   - Register forward hooks to capture layer outputs
   - Focus on residual stream at each layer (following persona vectors methodology)
   - Capture activations for all transformer layers

4. **Inference**
   - Process each interesting sample through model
   - Save activations: `[batch_size, sequence_length, hidden_dim]` per layer
   - Process each boring sample through model  
   - Save activations with same shape

5. **Vector Computation**
   - Average activations across 20 interesting samples → **A_interesting**
   - Average activations across 20 boring samples → **A_boring**
   - Compute difference: **tedium_vector = A_boring - A_interesting** (per layer)

6. **Analysis**
   - Compute L2 norm of tedium vector at each layer
   - Plot magnitude across layers (expect peak in middle layers)
   - Identify which layers show strongest signal

### Layer Selection

Following the persona vectors methodology:

1. **Magnitude**: Calculate L2 norm of tedium vector at each layer
   - Larger differences = stronger signal
   - Expect middle layers to show strongest effects

2. **Consistency** (future validation):
   - Run multiple trials with different interesting texts
   - Run multiple trials with different boring patterns
   - If same layers consistently show similar tedium vectors, that's evidence of robustness

3. **Steering Effectiveness** (future validation):
   - Extract tedium vectors from different layers
   - Try steering *away* from them during inference on repetitive tasks
   - Layer with biggest impact on output quality = where tedium matters most

### Output

- Tedium vectors saved per layer (for future steering experiments)
- Visualization: plot of tedium vector magnitude across layers
- Analysis: which layers show strongest signal

## Technical Considerations

### Context Length vs. Token Count

- Model has 128K max context length
- We only process actual tokens sent (4,096 per sample)
- No zero-padding to max length
- Activations shaped by actual sequence length, not theoretical maximum

### Quantization Effects

- 4-bit quantization may introduce noise
- This is actually a feature: if we detect tedium through quantization noise, it's a robust signal

### Computational Constraints

- 20 samples per condition = manageable
- Can run on Jeffery's laptop (already runs gpt-oss-20b)
- May take some time but not prohibitive

## Future Directions

1. **Steering Experiments**: Use extracted tedium vector to steer model *away* from tedium during repetitive tasks

2. **Third Condition**: Test random word lists (semantic noise) as distinct from both interesting and boring

3. **Robustness Testing**: Multiple trials with varied content to validate consistency

4. **Cross-Model Comparison**: Test whether tedium vectors transfer across different models

5. **Mechanistic Interpretation**: Connect findings to attention patterns and implicit weight updates

## Implementation Workflow

1. **Phase 1**: Write Python script to load model and capture activations
2. **Phase 2**: Add dataset loading (The Pile + synthetic boring text)  
3. **Phase 3**: Run inference and save activations
4. **Phase 4**: Compute tedium vectors and analyze
5. **Phase 5** (optional): Convert to Jupyter notebook for interactive exploration

## Open Questions

- Will 4-bit quantization wash out the signal or preserve it?
- Which layers will show the strongest tedium signal?
- Does tedium accumulate linearly with sequence length?
- Can we detect tedium through quantization noise?

## References

- Persona Vectors Paper (in project files)
- "Learning without training: The implicit dynamics of in-context learning" (Dherin et al., arXiv 2507.16003)
- Tim Kellogg's boredom experiment

---

**Status**: Design phase complete, ready to implement
**Next Step**: Write Python script for model loading and activation capture
