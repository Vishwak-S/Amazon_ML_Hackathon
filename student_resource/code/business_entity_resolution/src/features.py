"""
features.py — Pairwise feature engineering for entity resolution.

Computes string similarity features between business record pairs.
"""

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import numpy as np


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def containment_similarity(set_a: set, set_b: set) -> float:
    """What fraction of set_a is in set_b (asymmetric containment)."""
    if not set_a:
        return 1.0 if not set_b else 0.0
    intersection = len(set_a & set_b)
    return intersection / len(set_a)


def compute_pairwise_features(
    name1: str, name2: str,
    addr1: str, addr2: str,
    country1: str, country2: str,
    name_tokens1: set, name_tokens2: set,
    addr_tokens1: set, addr_tokens2: set,
    name_ngrams1: set, name_ngrams2: set,
    addr_ngrams1: set, addr_ngrams2: set,
) -> np.ndarray:
    """
    Compute feature vector for a pair of records.
    Returns numpy array of features.
    """
    features = []

    # ── Country match ────────────────────────────────────────────────────
    features.append(1.0 if country1 == country2 else 0.0)

    # ── Name features ────────────────────────────────────────────────────
    # Levenshtein ratio (normalized)
    if name1 or name2:
        features.append(fuzz.ratio(name1, name2) / 100.0)
    else:
        features.append(1.0)

    # Token sort ratio
    if name1 or name2:
        features.append(fuzz.token_sort_ratio(name1, name2) / 100.0)
    else:
        features.append(1.0)

    # Token set ratio
    if name1 or name2:
        features.append(fuzz.token_set_ratio(name1, name2) / 100.0)
    else:
        features.append(1.0)

    # Partial ratio (handles substring matching like DBA names)
    if name1 or name2:
        features.append(fuzz.partial_ratio(name1, name2) / 100.0)
    else:
        features.append(1.0)

    # Jaccard on name tokens
    features.append(jaccard_similarity(name_tokens1, name_tokens2))

    # Containment: how much of name1 tokens in name2, and vice versa
    features.append(containment_similarity(name_tokens1, name_tokens2))
    features.append(containment_similarity(name_tokens2, name_tokens1))

    # Character n-gram Jaccard on names
    features.append(jaccard_similarity(name_ngrams1, name_ngrams2))

    # Name lengths ratio
    len1, len2 = len(name1), len(name2)
    if max(len1, len2) > 0:
        features.append(min(len1, len2) / max(len1, len2))
    else:
        features.append(1.0)

    # Number of name tokens
    features.append(len(name_tokens1))
    features.append(len(name_tokens2))

    # ── Address features ─────────────────────────────────────────────────
    # Levenshtein ratio
    if addr1 or addr2:
        features.append(fuzz.ratio(addr1, addr2) / 100.0)
    else:
        features.append(1.0)

    # Token sort ratio
    if addr1 or addr2:
        features.append(fuzz.token_sort_ratio(addr1, addr2) / 100.0)
    else:
        features.append(1.0)

    # Token set ratio
    if addr1 or addr2:
        features.append(fuzz.token_set_ratio(addr1, addr2) / 100.0)
    else:
        features.append(1.0)

    # Partial ratio
    if addr1 or addr2:
        features.append(fuzz.partial_ratio(addr1, addr2) / 100.0)
    else:
        features.append(1.0)

    # Jaccard on address tokens
    features.append(jaccard_similarity(addr_tokens1, addr_tokens2))

    # Containment on address tokens
    features.append(containment_similarity(addr_tokens1, addr_tokens2))
    features.append(containment_similarity(addr_tokens2, addr_tokens1))

    # Character n-gram Jaccard on addresses
    features.append(jaccard_similarity(addr_ngrams1, addr_ngrams2))

    # Address lengths ratio
    len1, len2 = len(addr1), len(addr2)
    if max(len1, len2) > 0:
        features.append(min(len1, len2) / max(len1, len2))
    else:
        features.append(1.0)

    # Number of address tokens
    features.append(len(addr_tokens1))
    features.append(len(addr_tokens2))

    # ── Combined features ────────────────────────────────────────────────
    # Average of name + address similarities
    name_avg = (features[1] + features[2] + features[3]) / 3.0
    addr_avg = (features[12] + features[13] + features[14]) / 3.0
    features.append(name_avg)
    features.append(addr_avg)
    features.append(name_avg * addr_avg)  # interaction
    features.append(min(name_avg, addr_avg))  # min of the two

    # Number overlap in address (for street numbers / PIN codes)
    nums1 = set(t for t in (addr_tokens1 if addr_tokens1 else set()) if t.isdigit())
    nums2 = set(t for t in (addr_tokens2 if addr_tokens2 else set()) if t.isdigit())
    features.append(jaccard_similarity(nums1, nums2))

    # Whether both addresses have numbers
    features.append(1.0 if nums1 and nums2 else 0.0)

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    'country_match',
    'name_levenshtein', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_jaccard', 'name_contain_1in2', 'name_contain_2in1',
    'name_ngram_jaccard', 'name_len_ratio', 'name_ntokens1', 'name_ntokens2',
    'addr_levenshtein', 'addr_token_sort', 'addr_token_set', 'addr_partial',
    'addr_jaccard', 'addr_contain_1in2', 'addr_contain_2in1',
    'addr_ngram_jaccard', 'addr_len_ratio', 'addr_ntokens1', 'addr_ntokens2',
    'name_avg_sim', 'addr_avg_sim', 'name_addr_interaction', 'min_name_addr',
    'addr_num_jaccard', 'both_have_nums',
]

NUM_FEATURES = len(FEATURE_NAMES)
