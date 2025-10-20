"""
Prepare OneStopEnglish Pairs for Complexity Vector Extraction

Downloads the OneStopEnglish corpus and creates paired datasets:
- Elementary (simple) vs Advanced (complex)
- Same articles, different reading levels
- Parallel structure to wikipedia_pairs.json
"""

from datasets import load_dataset
import json
from pathlib import Path
import textstat
from tqdm import tqdm

OUTPUT_FILE = Path("./data/onestop_pairs/onestop_pairs.json")


def main():
    print("="*80)
    print("ONESTOP ENGLISH PAIRS PREPARATION")
    print("="*80)
    print("\nLoading OneStopEnglish dataset from HuggingFace...")

    ds = load_dataset('onestop_english')['train']

    print(f"✓ Loaded {len(ds)} texts")
    print(f"  - Elementary: {sum(1 for x in ds if x['label'] == 0)}")
    print(f"  - Intermediate: {sum(1 for x in ds if x['label'] == 1)}")
    print(f"  - Advanced: {sum(1 for x in ds if x['label'] == 2)}")

    # The dataset has 189 articles × 3 levels = 567 total
    # We need to group them into triplets
    # Assuming they're ordered: all elementary, then all intermediate, then all advanced

    elementary = [x['text'] for x in ds if x['label'] == 0]
    intermediate = [x['text'] for x in ds if x['label'] == 1]
    advanced = [x['text'] for x in ds if x['label'] == 2]

    num_articles = len(elementary)
    assert len(intermediate) == num_articles and len(advanced) == num_articles, \
        "Unequal number of texts at each level!"

    print(f"\n✓ Found {num_articles} article triplets")

    # Create pairs: elementary vs advanced (skip intermediate for now)
    print("\nComputing Flesch-Kincaid grades...")

    pairs = []
    for i in tqdm(range(num_articles), desc="Processing articles"):
        simple_text = elementary[i]
        complex_text = advanced[i]

        # Compute FK grades
        simple_fk = textstat.flesch_kincaid_grade(simple_text)
        complex_fk = textstat.flesch_kincaid_grade(complex_text)

        # Extract topic (first sentence, first 50 chars)
        topic = simple_text.split('.')[0][:50] + "..."

        pairs.append({
            "topic": f"Article {i+1}: {topic}",
            "simple_text": simple_text,
            "simple_grade": round(simple_fk, 1),
            "regular_text": complex_text,
            "regular_grade": round(complex_fk, 1),
        })

    # Summary statistics
    avg_simple = sum(p['simple_grade'] for p in pairs) / len(pairs)
    avg_complex = sum(p['regular_grade'] for p in pairs) / len(pairs)

    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Average FK grade (elementary): {avg_simple:.1f}")
    print(f"Average FK grade (advanced):   {avg_complex:.1f}")
    print(f"Average difference:            {avg_complex - avg_simple:.1f} grades")

    # Save
    output = {
        "source": "OneStopEnglish corpus",
        "description": "189 news articles, each at Elementary and Advanced reading levels",
        "pairs": pairs
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(pairs)} pairs to: {OUTPUT_FILE}")
    print(f"\n{'='*80}")
    print("✅ PREPARATION COMPLETE")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Run extract_complexity_vectors.py with DATA_PATH pointing to this file")
    print("  2. Compare results with Wikipedia-based extraction")


if __name__ == "__main__":
    main()
