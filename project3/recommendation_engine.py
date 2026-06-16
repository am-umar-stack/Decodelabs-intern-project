#!/usr/bin/env python3
"""
================================================================================
 ARTIFICIAL INTELLIGENCE PROJECT 3 — AI Recommendation Logic
================================================================================

 Author       : DecodeLabs — Personalization & Data Scientist Engineer
 Project      : Pattern Alignment — matching user profile vectors with item
                attribute vectors using mathematical similarity formulas.

 Core Problem :
   Given a user's interests (expressed as tags/features) and a catalogue of
   items (each described by tags/features), RANK the items by how well they
   align with the user's profile.

 Two Approaches (from simple → advanced):
   ┌──────────────────────┐      ┌──────────────────────────────────┐
   │  APPROACH 1:          │      │  APPROACH 2:                      │
   │  Binary Overlap       │      │  TF-IDF Weighted                  │
   │  (Jaccard Similarity) │      │  (Cosine Similarity)              │
   │                       │      │                                    │
   │  • Presence/absence   │      │  • Penalises generic tags          │
   │    only (1s and 0s)   │      │  • Rewards specific tag alignment  │
   │  • Treats all tags    │      │  • Weights by inverse document     │
   │    equally            │      │    frequency (rarity)              │
   │  • Simple set math    │      │  • Proper vector space model       │
   └──────────────────────┘      └──────────────────────────────────┘

 Shared Requirement:
   ╔═══════════════════════════════════════════════════════════════════╗
   ║  UNIFIED VOCABULARY INDEX                                        ║
   ║  Both user profiles and item attributes MUST map to the same     ║
   ║  vocabulary dictionary. If "sci-fi" is index 5 for a user,      ║
   ║  it must also be index 5 for every item. Misalignment = broken   ║
   ║  math and garbage recommendations.                               ║
   ╚═══════════════════════════════════════════════════════════════════╝

================================================================================
"""

import numpy as np
from collections import Counter
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA — Users and Items
# ─────────────────────────────────────────────────────────────────────────────

# Each user has a "profile" expressed as a list of interest tags.
# Each item has "attributes" expressed as a list of descriptive tags.
# Tags overlap naturally — some tags are common (e.g., "action", "comedy"),
# while others are rare and specific (e.g., "cyberpunk", "time-travel").

USERS: dict[str, list[str]] = {
    "Alice": [
        "sci-fi", "space", "dystopia", "artificial-intelligence",
        "cyberpunk", "thriller",
    ],
    "Bob": [
        "comedy", "romance", "feel-good", "coming-of-age",
        "friendship",
    ],
    "Charlie": [
        "action", "adventure", "fantasy", "magic",
        "epic", "medieval",
    ],
    "Diana": [
        "mystery", "crime", "detective", "psychological",
        "noir", "thriller",
    ],
}

ITEMS: dict[str, dict] = {
    "Blade Runner 2049": {
        "tags": [
            "sci-fi", "dystopia", "artificial-intelligence",
            "cyberpunk", "thriller", "philosophical", "noir",
        ],
        "description": "A sci-fi noir set in a dystopian future with AI.",
    },
    "The Martian": {
        "tags": [
            "sci-fi", "space", "survival", "adventure",
            "humor", "realistic",
        ],
        "description": "An astronaut stranded on Mars uses science to survive.",
    },
    "Interstellar": {
        "tags": [
            "sci-fi", "space", "dystopia", "time-travel",
            "emotional", "epic", "philosophical",
        ],
        "description": "A space epic about love, time, and survival.",
    },
    "The Notebook": {
        "tags": [
            "romance", "emotional", "coming-of-age",
            "love-story", "classic",
        ],
        "description": "A timeless love story across decades.",
    },
    "Superbad": {
        "tags": [
            "comedy", "coming-of-age", "friendship",
            "high-school", "feel-good", "humor",
        ],
        "description": "Teen friends navigate a wild night before graduation.",
    },
    "The Lord of the Rings": {
        "tags": [
            "fantasy", "adventure", "epic", "magic",
            "medieval", "friendship", "quest",
        ],
        "description": "An epic fantasy quest to destroy a powerful ring.",
    },
    "Sherlock Holmes": {
        "tags": [
            "mystery", "crime", "detective", "thriller",
            "Victorian", "clever",
        ],
        "description": "The brilliant detective solves London's darkest crimes.",
    },
    "Inception": {
        "tags": [
            "sci-fi", "thriller", "psychological",
            "action", "dream", "mind-bending",
        ],
        "description": "A thief who steals secrets through dream infiltration.",
    },
    "Harry Potter": {
        "tags": [
            "fantasy", "magic", "coming-of-age", "friendship",
            "adventure", "school",
        ],
        "description": "A young wizard's journey at a magical school.",
    },
    "Breaking Bad": {
        "tags": [
            "crime", "thriller", "dystopia", "psychological",
            "dark", "intense",
        ],
        "description": "A chemistry teacher turns to a life of crime.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — VOCABULARY INDEX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_vocabulary(
    users: dict[str, list[str]],
    items: dict[str, dict],
) -> tuple[dict[str, int], list[str]]:
    """
    Build a UNIFIED vocabulary index from ALL tags across ALL users and items.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  WHY A SHARED VOCABULARY INDEX IS CRITICAL                       ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  Both user vectors and item vectors must be encoded using the    ║
    ║  SAME dictionary. If "sci-fi" maps to index 3 for users, it     ║
    ║  MUST map to index 3 for items. Otherwise, you'd be comparing   ║
    ║  unrelated dimensions — like comparing height to weight.        ║
    ║                                                                   ║
    ║  This function:                                                  ║
    ║    1. Collects every unique tag from users AND items             ║
    ║    2. Sorts them alphabetically (deterministic ordering)         ║
    ║    3. Assigns each tag a stable integer index [0, 1, 2, ...]    ║
    ║    4. Returns the tag→index map and the index→tag list          ║
    ║                                                                   ║
    ║  Result: Every vector in the system has the same length (= the   ║
    ║  total number of unique tags), and dimension k always means      ║
    ║  the same tag.                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        users : dict mapping username → list of interest tags.
        items : dict mapping item name  → dict with 'tags' key.

    Returns:
        tag_to_index : dict mapping each tag string → integer index.
        index_to_tag : list where position i holds the tag string.
    """
    # Collect ALL unique tags from both users and items
    all_tags: set[str] = set()

    for user_tags in users.values():
        all_tags.update(user_tags)

    for item_data in items.values():
        all_tags.update(item_data["tags"])

    # Sort alphabetically for deterministic, reproducible ordering
    sorted_tags = sorted(all_tags)

    # Build the bidirectional mapping
    tag_to_index = {tag: idx for idx, tag in enumerate(sorted_tags)}
    index_to_tag = sorted_tags

    return tag_to_index, index_to_tag


def tags_to_binary_vector(
    tags: list[str],
    tag_to_index: dict[str, int],
) -> np.ndarray:
    """
    Convert a list of tags into a binary (0/1) presence vector.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  BINARY VECTOR ENCODING                                           ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  Vocabulary: ["action", "comedy", "fantasy", "romance", "sci-fi"] ║
    ║  Indices:      0         1         2         3         4         ║
    ║                                                                   ║
    ║  User tags: ["comedy", "sci-fi"]                                  ║
    ║  Binary vector: [0, 1, 0, 0, 1]                                  ║
    ║                                                                   ║
    ║  • 1 = tag is PRESENT in the profile/item                        ║
    ║  • 0 = tag is ABSENT                                              ║
    ║                                                                   ║
    ║  Limitation: This treats "comedy" (appears in 5 items) the same  ║
    ║  as "cyberpunk" (appears in 1 item). No notion of specificity.   ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        tags        : List of tag strings (e.g., ["sci-fi", "thriller"]).
        tag_to_index: Unified tag → index mapping from build_vocabulary().

    Returns:
        Binary numpy array of length |vocabulary| with 1s at tag positions.
    """
    vocab_size = len(tag_to_index)
    vector = np.zeros(vocab_size, dtype=np.float64)

    for tag in tags:
        if tag in tag_to_index:
            vector[tag_to_index[tag]] = 1.0

    return vector


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — JACCARD SIMILARITY (Binary Overlap)
# ══════════════════════════════════════════════════════════════════════════════

def jaccard_similarity(
    set_a: set[str],
    set_b: set[str],
) -> float:
    """
    Compute the Jaccard Similarity Coefficient between two sets.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  THE MATH BEHIND JACCARD SIMILARITY                              ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  DEFINITION:                                                      ║
    ║                    | A ∩ B |                                      ║
    ║    J(A, B)  =  ───────────────                                   ║
    ║                   | A ∪ B |                                      ║
    ║                                                                   ║
    ║  Where:                                                           ║
    ║    |A ∩ B| = number of elements in BOTH A and B (intersection)    ║
    ║    |A ∪ B| = number of elements in EITHER A or B (union)          ║
    ║                                                                   ║
    ║  RANGE: [0.0, 1.0]                                               ║
    ║    • 0.0 = no overlap at all                                      ║
    ║    • 1.0 = identical sets                                         ║
    ║                                                                   ║
    ║  EXAMPLE:                                                         ║
    ║    User tags: {sci-fi, space, dystopia, thriller}                 ║
    ║    Item tags: {sci-fi, dystopia, thriller, philosophical}         ║
    ║                                                                   ║
    ║    Intersection: {sci-fi, dystopia, thriller}         → |3|      ║
    ║    Union:        {sci-fi, space, dystopia, thriller,  → |5|      ║
    ║                   philosophical}                                  ║
    ║                                                                   ║
    ║    J = 3/5 = 0.600                                               ║
    ║                                                                   ║
    ║  EQUIVALENCE TO BINARY VECTORS:                                   ║
    ║    With binary encoding [0,1,0,1,1,1,0,...] for both,            ║
    ║    Jaccard = (A · B) / (|A|² + |B|² - A · B)                     ║
    ║    It's the ratio of co-activated bits to any activated bit.     ║
    ║                                                                   ║
    ║  LIMITATION:                                                      ║
    ║    All tags treated equally. "action" (generic, appears in many   ║
    ║    items) carries the same weight as "cyberpunk" (specific,       ║
    ║    appears in 1 item). This is where Approach 2 improves.        ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        set_a: First set of tags (e.g., user interests).
        set_b: Second set of tags (e.g., item attributes).

    Returns:
        Jaccard similarity coefficient in [0.0, 1.0].
    """
    intersection = set_a & set_b      # Tags in BOTH sets
    union        = set_a | set_b      # Tags in EITHER set

    # Edge case: if both sets are empty, define Jaccard as 0.0
    # (no meaningful overlap to report)
    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def jaccard_ranking(
    user_name: str,
    user_tags: list[str],
    items: dict[str, dict],
    tag_to_index: dict[str, int],
    top_n: Optional[int] = None,
) -> list[tuple[str, float, set[str]]]:
    """
    Rank ALL items by Jaccard similarity to a user's profile.

    Args:
        user_name   : Display name of the user.
        user_tags   : List of the user's interest tags.
        items       : Item catalogue.
        tag_to_index: Unified vocabulary mapping.
        top_n       : If set, return only the top N results.

    Returns:
        Sorted list of (item_name, jaccard_score, overlap_tags).
    """
    user_set = set(user_tags)
    results = []

    for item_name, item_data in items.items():
        item_set = set(item_data["tags"])
        score = jaccard_similarity(user_set, item_set)
        overlap = user_set & item_set
        results.append((item_name, score, overlap))

    # Sort by score descending (highest similarity first)
    results.sort(key=lambda x: x[1], reverse=True)

    if top_n:
        results = results[:top_n]

    return results


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — TF-IDF WEIGHTED COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_idf(
    items: dict[str, dict],
    tag_to_index: dict[str, int],
) -> np.ndarray:
    """
    Compute the Inverse Document Frequency (IDF) for every tag in the vocabulary.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  THE MATH BEHIND TF-IDF (Term Frequency – Inverse Doc Frequency) ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  INTUITION:                                                       ║
    ║    A tag that appears in EVERY item (e.g., "action") tells us    ║
    ║    nothing about what makes an item SPECIAL. It's generic.       ║
    ║    A tag that appears in only 1 item (e.g., "cyberpunk") is      ║
    ║    highly DISCRIMINATING — matching on it means real alignment.  ║
    ║                                                                   ║
    ║  IDF FORMULA:                                                     ║
    ║                     N                                             ║
    ║    IDF(t) = log ( ─────── ) + 1                                  ║
    ║                   df(t)                                           ║
    ║                                                                   ║
    ║  Where:                                                           ║
    ║    N     = total number of items (documents)                      ║
    ║    df(t) = number of items that contain tag t                     ║
    ║    log   = natural logarithm (base e)                             ║
    ║    +1    = smoothing to avoid division by zero                    ║
    ║                                                                   ║
    ║  EXAMPLE (with 10 items):                                         ║
    ║    "action"  appears in 2 items → IDF = log(10/2)+1 = 2.609      ║
    ║    "sci-fi"  appears in 4 items → IDF = log(10/4)+1 = 1.916      ║
    ║    "fantasy" appears in 2 items → IDF = log(10/2)+1 = 2.609      ║
    ║    "thriller"appears in 4 items → IDF = log(10/4)+1 = 1.916      ║
    ║                                                                   ║
    ║  Result: Generic tags get LOW weight, specific tags get HIGH     ║
    ║  weight. This is the KEY improvement over binary (Approach 1).   ║
    ║                                                                   ║
    ║  TF (Term Frequency) for our use case:                           ║
    ║    Since tags are either present (1) or absent (0), TF = 1 for   ║
    ║    any present tag. The final weight = TF × IDF = 1 × IDF = IDF. ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        items       : Item catalogue.
        tag_to_index: Unified vocabulary mapping.

    Returns:
        IDF array of shape (vocab_size,) with weight for each tag index.
    """
    vocab_size = len(tag_to_index)
    num_items = len(items)

    # Count how many items contain each tag (document frequency)
    doc_frequency = np.zeros(vocab_size, dtype=np.float64)

    for item_data in items.values():
        for tag in item_data["tags"]:
            if tag in tag_to_index:
                doc_frequency[tag_to_index[tag]] += 1.0

    # Compute IDF: log(N / df(t)) + 1
    # Add 1 to df to prevent division by zero (Laplace smoothing)
    idf = np.log(num_items / (doc_frequency + 1)) + 1.0

    return idf


def tags_to_tfidf_vector(
    tags: list[str],
    tag_to_index: dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    """
    Convert a list of tags into a TF-IDF weighted vector.

    Since each tag appears at most once (binary TF), the final weight
    for each present tag is simply its IDF value.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  TF-IDF VECTOR CONSTRUCTION                                      ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  For tag t at index j:                                            ║
    ║    • If tag is PRESENT:  vector[j] = TF(t) × IDF(t) = 1 × IDF(t)║
    ║    • If tag is ABSENT:   vector[j] = 0                           ║
    ║                                                                   ║
    ║  Example (simplified vocabulary):                                  ║
    ║    Vocab:  ["action", "cyberpunk", "fantasy", "sci-fi", "thriller"]║
    ║    IDF:    [2.609,     3.303,       2.609,     1.916,     1.916]  ║
    ║                                                                   ║
    ║    Tags: ["sci-fi", "cyberpunk", "thriller"]                      ║
    ║    Vector: [0, 3.303, 0, 1.916, 1.916]                           ║
    ║                                                                   ║
    ║    "cyberpunk" (rare) gets weight 3.303 (high reward for match)  ║
    ║    "sci-fi" (common) gets weight 1.916 (lower reward for match)  ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        tags        : List of tag strings.
        tag_to_index: Unified vocabulary mapping.
        idf         : Pre-computed IDF array.

    Returns:
        TF-IDF weighted vector of shape (vocab_size,).
    """
    vocab_size = len(tag_to_index)
    vector = np.zeros(vocab_size, dtype=np.float64)

    for tag in tags:
        if tag in tag_to_index:
            idx = tag_to_index[tag]
            # TF = 1 (binary: tag is present), so weight = 1 × IDF
            vector[idx] = 1.0 * idf[idx]

    return vector


def cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """
    Compute the Cosine Similarity between two vectors.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  THE MATH BEHIND COSINE SIMILARITY                               ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  DEFINITION:                                                      ║
    ║                       A · B                                       ║
    ║    cos(θ)  =  ─────────────────────                               ║
    ║               ||A|| × ||B||                                       ║
    ║                                                                   ║
    ║  Where:                                                           ║
    ║    A · B  = dot product = Σ(A_i × B_i)                           ║
    ║    ||A||  = L2 norm    = sqrt(Σ A_i²)                            ║
    ║    ||B||  = L2 norm    = sqrt(Σ B_i²)                            ║
    ║                                                                   ║
    ║  RANGE: [0.0, 1.0] for non-negative vectors                     ║
    ║    • 0.0 = vectors are orthogonal (no shared direction)          ║
    ║    • 1.0 = vectors are identical (perfect alignment)             ║
    ║                                                                   ║
    ║  WHY COSINE (not Euclidean)?                                     ║
    ║    Cosine measures DIRECTIONAL alignment, ignoring magnitude.    ║
    ║    Two vectors pointing the same way but with different lengths  ║
    ║    have cosine similarity = 1.0. This is ideal for TF-IDF       ║
    ║    vectors where we care about WHICH features match, not HOW    ║
    ║    MANY total features a profile/item has.                       ║
    ║                                                                   ║
    ║  EXAMPLE:                                                         ║
    ║    User vec:   [0, 3.303, 0, 1.916, 1.916, 0, 0, ...]           ║
    ║    Item vec:   [0, 3.303, 0, 1.916, 1.916, 0, 0, ...]           ║
    ║    → Perfect match → cos(θ) = 1.0                                ║
    ║                                                                   ║
    ║    User vec:   [0, 0, 0, 1.916, 0, 0, 0, ...]                   ║
    ║    Item vec:   [2.609, 0, 0, 0, 0, 1.916, 0, ...]               ║
    ║    → No shared tags → cos(θ) = 0.0                               ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Args:
        vec_a: First vector  (e.g., user TF-IDF vector).
        vec_b: Second vector (e.g., item TF-IDF vector).

    Returns:
        Cosine similarity in [0.0, 1.0] for non-negative vectors.
    """
    dot_product = np.dot(vec_a, vec_b)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Edge case: if either vector is zero, similarity is undefined → return 0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def tfidf_cosine_ranking(
    user_name: str,
    user_tags: list[str],
    items: dict[str, dict],
    tag_to_index: dict[str, int],
    idf: np.ndarray,
    top_n: Optional[int] = None,
) -> list[tuple[str, float, list[str]]]:
    """
    Rank ALL items by TF-IDF weighted Cosine Similarity to a user's profile.

    Args:
        user_name   : Display name of the user.
        user_tags   : List of the user's interest tags.
        items       : Item catalogue.
        tag_to_index: Unified vocabulary mapping.
        idf         : Pre-computed IDF weights.
        top_n       : If set, return only the top N results.

    Returns:
        Sorted list of (item_name, cosine_score, matching_tags).
    """
    # Build the user's TF-IDF vector
    user_vector = tags_to_tfidf_vector(user_tags, tag_to_index, idf)

    results = []

    for item_name, item_data in items.items():
        # Build the item's TF-IDF vector
        item_vector = tags_to_tfidf_vector(item_data["tags"], tag_to_index, idf)

        # Compute cosine similarity
        score = cosine_similarity(user_vector, item_vector)

        # Find which tags actually overlap (for display)
        matching = sorted(set(user_tags) & set(item_data["tags"]))

        results.append((item_name, score, matching))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    if top_n:
        results = results[:top_n]

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_vocabulary_info(
    tag_to_index: dict[str, int],
    idf: np.ndarray,
) -> None:
    """Display the unified vocabulary and per-tag IDF weights."""
    print("\n" + "=" * 76)
    print("  UNIFIED VOCABULARY INDEX")
    print("=" * 76)
    print(f"\n  Total unique tags: {len(tag_to_index)}\n")
    print(f"  {'Index':>6}  {'Tag':<30}  {'IDF Weight':>12}  {'Specificity':>12}")
    print(f"  {'─'*6}  {'─'*30}  {'─'*12}  {'─'*12}")

    # Sort by index for display
    sorted_items = sorted(tag_to_index.items(), key=lambda x: x[1])
    for tag, idx in sorted_items:
        weight = idf[idx]
        # Classify specificity based on IDF weight
        if weight >= 3.0:
            spec = "🔵 HIGH"
        elif weight >= 2.0:
            spec = "🟢 MED"
        else:
            spec = "🔴 LOW"
        print(f"  {idx:>6}  {tag:<30}  {weight:>12.4f}  {spec:>12}")


def print_jaccard_results(
    user_name: str,
    results: list[tuple[str, float, set[str]]],
) -> None:
    """Pretty-print Jaccard similarity results."""
    print(f"\n  🎯 Top recommendations for {user_name} (Jaccard):")
    print(f"  {'Rank':>5}  {'Score':>8}  {'Item':<30}  {'Overlapping Tags'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*30}  {'─'*30}")
    for rank, (item, score, overlap) in enumerate(results, 1):
        overlap_str = ", ".join(sorted(overlap)) if overlap else "(none)"
        bar = "█" * int(score * 20)
        print(f"  {rank:>5}  {score:>8.4f}  {item:<30}  {bar} {overlap_str}")


def print_tfidf_results(
    user_name: str,
    results: list[tuple[str, float, list[str]]],
) -> None:
    """Pretty-print TF-IDF Cosine Similarity results."""
    print(f"\n  🎯 Top recommendations for {user_name} (TF-IDF Cosine):")
    print(f"  {'Rank':>5}  {'Score':>8}  {'Item':<30}  {'Matching Tags'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*30}  {'─'*30}")
    for rank, (item, score, matching) in enumerate(results, 1):
        match_str = ", ".join(matching) if matching else "(none)"
        bar = "█" * int(score * 20)
        print(f"  {rank:>5}  {score:>8.4f}  {item:<30}  {bar} {match_str}")


def print_method_comparison(
    user_name: str,
    jaccard_results: list[tuple[str, float, set[str]]],
    tfidf_results: list[tuple[str, float, list[str]]],
) -> None:
    """Show side-by-side comparison of both methods for one user."""
    print(f"\n  {'═'*72}")
    print(f"  📊 COMPARISON: Jaccard vs TF-IDF Cosine for {user_name}")
    print(f"  {'═'*72}")
    print(f"\n  {'Rank':>5}  {'Jaccard':>10}  {'TF-IDF':>10}  {'Item':<30}  {'Difference'}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*30}  {'─'*12}")

    jaccard_dict = {item: score for item, score, _ in jaccard_results}
    tfidf_dict = {item: score for item, score, _ in tfidf_results}

    for rank, (item, tfidf_score, _) in enumerate(tfidf_results[:6], 1):
        jaccard_score = jaccard_dict.get(item, 0.0)
        diff = tfidf_score - jaccard_score
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {rank:>5}  {jaccard_score:>10.4f}  {tfidf_score:>10.4f}  "
              f"{item:<30}  {arrow} {abs(diff):.4f}")

    # Highlight the biggest rank changes
    jaccard_rank_map = {item: r for r, (item, _, _) in enumerate(jaccard_results, 1)}
    tfidf_rank_map = {item: r for r, (item, _, _) in enumerate(tfidf_results, 1)}

    print(f"\n  🔄 Biggest rank changes (Jaccard → TF-IDF Cosine):")
    changes = []
    for item in jaccard_rank_map:
        j_rank = jaccard_rank_map[item]
        t_rank = tfidf_rank_map.get(item, len(tfidf_results))
        if j_rank != t_rank:
            changes.append((item, j_rank, t_rank, j_rank - t_rank))

    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    for item, j_rank, t_rank, delta in changes[:5]:
        direction = "⬆ promoted" if delta > 0 else "⬇ demoted"
        print(f"     {item:<30}  #{j_rank} → #{t_rank}  ({direction} by {abs(delta)})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — ORCHESTRATE THE FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full recommendation pipeline."""

    print()
    print("╔" + "═" * 74 + "╗")
    print("║" + " AI PROJECT 3 — Recommendation Logic: Pattern Alignment".center(74) + "║")
    print("║" + " DecodeLabs • Binary Overlap vs TF-IDF Weighted Matching".center(74) + "║")
    print("╚" + "═" * 74 + "╝")

    # ── STEP 0: Build Unified Vocabulary ─────────────────────────────
    tag_to_index, index_to_tag = build_vocabulary(USERS, ITEMS)
    idf = compute_idf(ITEMS, tag_to_index)
    print_vocabulary_info(tag_to_index, idf)

    # ── Show binary encoding for one user ────────────────────────────
    print("\n" + "=" * 76)
    print("  BINARY VECTOR ENCODING DEMO")
    print("=" * 76)
    alice_tags = USERS["Alice"]
    alice_binary = tags_to_binary_vector(alice_tags, tag_to_index)
    print(f"\n  Alice's tags: {alice_tags}")
    print(f"  Binary vector (showing non-zero positions):\n")
    for i, val in enumerate(alice_binary):
        if val == 1.0:
            print(f"    index {i:>2} → {index_to_tag[i]:<30} = {int(val)}")
    print(f"  Vector length: {len(alice_binary)} dimensions")

    # ── STEP 1: Jaccard Similarity (Approach 1) ──────────────────────
    print("\n" + "=" * 76)
    print("  APPROACH 1 — JACCARD SIMILARITY (Binary Overlap)")
    print("=" * 76)

    all_jaccard_results = {}
    for user_name, user_tags in USERS.items():
        results = jaccard_ranking(user_name, user_tags, ITEMS, tag_to_index)
        all_jaccard_results[user_name] = results
        print_jaccard_results(user_name, results)

    # ── STEP 2: TF-IDF Cosine Similarity (Approach 2) ────────────────
    print("\n" + "=" * 76)
    print("  APPROACH 2 — TF-IDF WEIGHTED COSINE SIMILARITY")
    print("=" * 76)

    all_tfidf_results = {}
    for user_name, user_tags in USERS.items():
        results = tfidf_cosine_ranking(
            user_name, user_tags, ITEMS, tag_to_index, idf,
        )
        all_tfidf_results[user_name] = results
        print_tfidf_results(user_name, results)

    # ── STEP 3: Side-by-Side Comparison ──────────────────────────────
    print("\n" + "=" * 76)
    print("  SIDE-BY-SIDE COMPARISON: Jaccard vs TF-IDF Cosine")
    print("=" * 76)

    for user_name in USERS:
        print_method_comparison(
            user_name,
            all_jaccard_results[user_name],
            all_tfidf_results[user_name],
        )

    # ── SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  ✅ PIPELINE COMPLETE — KEY TAKEAWAYS")
    print("=" * 76)
    print("""
  1. VOCABULARY ALIGNMENT:
     All users and items share the SAME tag→index mapping.
     This prevents dimensional misalignment in all vector operations.

  2. JACCARD (Approach 1):
     • Simple set-based intersection/union ratio.
     • Treats ALL tags equally — "action" (common) = "cyberpunk" (rare).
     • Good for quick, interpretable similarity checks.

  3. TF-IDF COSINE (Approach 2):
     • Penalises generic tags (low IDF = low weight).
     • Rewards specific tag matches (high IDF = high weight).
     • Measures directional alignment, not magnitude.
     • Produces RANKING SHIFTS — rare-tag matches rise to the top.

  4. WHEN TO USE WHICH:
     • Jaccard: Small catalogues, quick prototyping, equal tag importance.
     • TF-IDF Cosine: Production systems, large catalogues, real specificity.
""")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
