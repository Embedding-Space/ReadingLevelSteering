"""
Wikipedia Article Pair Extraction for Complexity Vector Training

Fetches matched pairs of Simple Wikipedia and Regular Wikipedia articles on the
same topics to create controlled training data for complexity vector extraction.

This controls for semantic content while isolating syntactic/structural complexity,
allowing us to extract a cleaner reading-level steering vector.

Experimental design:
- 20 topic pairs (Simple Wikipedia + Regular Wikipedia)
- Same topics ensure semantic similarity
- Complexity difference comes from writing style only
- Validates grade levels before accepting pairs
- Truncates to 4096 tokens for model processing
"""

import requests
import json
from pathlib import Path
import textstat
from typing import Tuple, Optional, List
from tqdm import tqdm


# Configuration
OUTPUT_DIR = Path("./data/wikipedia_pairs")
TARGET_TOKENS = 4096
MIN_SIMPLE_GRADE = 7.0
MAX_SIMPLE_GRADE = 11.0
MIN_COMPLEX_GRADE = 11.0
MAX_COMPLEX_GRADE = 17.0

# Article topics to fetch
# Selected for having good Simple Wikipedia coverage and substantive content
# Prioritizing concrete, natural phenomena and common concepts
TOPICS = [
    # Already successful (from first run)
    "Photosynthesis",
    "Solar System",
    "DNA",
    "World War II",
    "Ancient Rome",
    "Renaissance",
    "Internet",
    "Computer",
    "Electricity",
    "Climate change",
    "Lion",
    "Dolphin",
    "Evolution",
    "Albert Einstein",
    "William Shakespeare",
    "Democracy",
    "Music",
    "Mathematics",
    "Geography",
    "Human body",
    # Additional topics - simpler, concrete subjects
    "Earth",
    "Sun",
    "Moon",
    "Water",
    "Fire",
    "Rain",
    "Wind",
    "Tree",
    "Bacteria",
    "Bird",
    "Fish",
    "Dog",
    "Cat",
    "Food",
    "Agriculture",
    "Ocean",
    "Mountain",
    "River",
    "Weather",
    "Gravity",
]


def fetch_wikipedia_text(title: str, simple: bool = False) -> Optional[str]:
    """
    Fetch plain text content from Wikipedia article.

    Args:
        title: Article title
        simple: If True, fetch from Simple Wikipedia; otherwise Regular Wikipedia

    Returns:
        Plain text content or None if fetch fails
    """
    base_url = "https://simple.wikipedia.org" if simple else "https://en.wikipedia.org"
    api_url = f"{base_url}/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
    }

    # Add User-Agent header to avoid 403 Forbidden
    headers = {
        "User-Agent": "TediumVectorTinker/1.0 (Educational Research Project; https://github.com/Embedding-Space/TediumVectorTinker)"
    }

    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract text from response
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        # Get first (and should be only) page
        page = next(iter(pages.values()))

        # Check if page exists
        if "missing" in page:
            return None

        text = page.get("extract", "")
        return text if text else None

    except Exception as e:
        print(f"  ⚠ Error fetching {title} from {'Simple' if simple else 'Regular'} Wikipedia: {e}")
        return None


def truncate_to_token_count(text: str, target_tokens: int) -> str:
    """
    Truncate text to approximately target token count.

    Uses simple heuristic: ~1.3 tokens per word for English text.
    This is approximate but good enough for our purposes.
    """
    words = text.split()
    target_words = int(target_tokens / 1.3)

    if len(words) <= target_words:
        return text

    # Truncate to target words
    truncated_words = words[:target_words]
    return " ".join(truncated_words)


def validate_grade_level(text: str, is_simple: bool) -> Tuple[bool, float]:
    """
    Check if text falls in target grade level range.

    Args:
        text: Text to validate
        is_simple: If True, check against simple range; otherwise complex range

    Returns:
        Tuple of (is_valid, actual_grade_level)
    """
    try:
        grade = textstat.flesch_kincaid_grade(text)

        if is_simple:
            is_valid = MIN_SIMPLE_GRADE <= grade <= MAX_SIMPLE_GRADE
        else:
            is_valid = MIN_COMPLEX_GRADE <= grade <= MAX_COMPLEX_GRADE

        return is_valid, grade

    except:
        return False, 0.0


def fetch_and_validate_pair(topic: str) -> Optional[Tuple[str, str, float, float]]:
    """
    Fetch and validate a Simple/Regular Wikipedia pair.

    Args:
        topic: Article topic title

    Returns:
        Tuple of (simple_text, regular_text, simple_grade, regular_grade) or None if invalid
    """
    print(f"\nFetching: {topic}")

    # Fetch both versions
    simple_text = fetch_wikipedia_text(topic, simple=True)
    regular_text = fetch_wikipedia_text(topic, simple=False)

    if not simple_text or not regular_text:
        print(f"  ✗ Missing article(s)")
        return None

    # Truncate to target length
    simple_text = truncate_to_token_count(simple_text, TARGET_TOKENS)
    regular_text = truncate_to_token_count(regular_text, TARGET_TOKENS)

    # Validate grade levels
    simple_valid, simple_grade = validate_grade_level(simple_text, is_simple=True)
    regular_valid, regular_grade = validate_grade_level(regular_text, is_simple=False)

    print(f"  Simple grade: {simple_grade:.1f} {'✓' if simple_valid else '✗ (out of range)'}")
    print(f"  Regular grade: {regular_grade:.1f} {'✓' if regular_valid else '✗ (out of range)'}")

    if not simple_valid or not regular_valid:
        print(f"  ✗ Grade levels out of range")
        return None

    print(f"  ✓ Valid pair")
    return simple_text, regular_text, simple_grade, regular_grade


def save_pairs(pairs: List[Tuple[str, str, str, float, float]]):
    """
    Save validated pairs to JSON file.

    Args:
        pairs: List of (topic, simple_text, regular_text, simple_grade, regular_grade)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "num_pairs": len(pairs),
            "target_tokens": TARGET_TOKENS,
            "simple_grade_range": [MIN_SIMPLE_GRADE, MAX_SIMPLE_GRADE],
            "complex_grade_range": [MIN_COMPLEX_GRADE, MAX_COMPLEX_GRADE],
        },
        "pairs": [
            {
                "topic": topic,
                "simple_text": simple_text,
                "simple_grade": simple_grade,
                "regular_text": regular_text,
                "regular_grade": regular_grade,
            }
            for topic, simple_text, regular_text, simple_grade, regular_grade in pairs
        ]
    }

    output_path = OUTPUT_DIR / "wikipedia_pairs.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(pairs)} pairs to {output_path}")


def main():
    """Fetch and validate Wikipedia article pairs."""
    print("="*80)
    print("WIKIPEDIA ARTICLE PAIR EXTRACTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Target pairs: {len(TOPICS)}")
    print(f"  - Target tokens per article: {TARGET_TOKENS}")
    print(f"  - Simple grade range: {MIN_SIMPLE_GRADE} - {MAX_SIMPLE_GRADE}")
    print(f"  - Complex grade range: {MIN_COMPLEX_GRADE} - {MAX_COMPLEX_GRADE}")
    print(f"\nTopics to fetch:")
    for i, topic in enumerate(TOPICS, 1):
        print(f"  {i:2d}. {topic}")

    # Fetch and validate pairs
    valid_pairs = []

    print(f"\n{'='*80}")
    print("FETCHING AND VALIDATING PAIRS")
    print(f"{'='*80}")

    for topic in TOPICS:
        result = fetch_and_validate_pair(topic)
        if result:
            simple_text, regular_text, simple_grade, regular_grade = result
            valid_pairs.append((topic, simple_text, regular_text, simple_grade, regular_grade))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nValid pairs collected: {len(valid_pairs)} / {len(TOPICS)}")

    if valid_pairs:
        simple_grades = [g for _, _, _, g, _ in valid_pairs]
        regular_grades = [g for _, _, _, _, g in valid_pairs]

        print(f"\nSimple Wikipedia statistics:")
        print(f"  - Grade range: {min(simple_grades):.1f} - {max(simple_grades):.1f}")
        print(f"  - Mean grade: {sum(simple_grades)/len(simple_grades):.1f}")

        print(f"\nRegular Wikipedia statistics:")
        print(f"  - Grade range: {min(regular_grades):.1f} - {max(regular_grades):.1f}")
        print(f"  - Mean grade: {sum(regular_grades)/len(regular_grades):.1f}")

        # Save results
        save_pairs(valid_pairs)

        print(f"\n{'='*80}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"\nReady to extract complexity vectors using {len(valid_pairs)} topic pairs!")
        print(f"\nNext step: Run extract_complexity_vectors.py")
    else:
        print("\n⚠ No valid pairs found. Check grade level ranges or topic selection.")

    print()


if __name__ == "__main__":
    main()
