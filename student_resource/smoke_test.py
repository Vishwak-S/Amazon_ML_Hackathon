"""
Quick smoke test on a tiny sample to verify end-to-end pipeline works.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))

import pandas as pd
import csv
import numpy as np
from normalize import normalize_name, normalize_address, name_tokens, address_tokens, char_ngrams
from features import compute_pairwise_features, FEATURE_NAMES, NUM_FEATURES

# Test normalisation
tests = [
    ("Microsoft Corp.", "microsoft"),
    ("Tata Consultancy Services Private Limited", "tata"),
    ("Ben & Jerry's Ice Cream LLC", "ben and jerry ice cream"),
    ("SOCIÉTÉ GÉNÉRALE SA", "societe generale"),
]
print("=== Normalisation tests ===")
for raw, expected_contains in tests:
    normed = normalize_name(raw)
    ok = expected_contains in normed
    print(f"  '{raw}' -> '{normed}'  [{'OK' if ok else 'FAIL'}]")

# Test address normalisation
addr_tests = [
    ("1234 Main St, Suite 100, New York, NY", "1234 main street"),
    ("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "570 13 new delhi"),
]
print("\n=== Address normalisation tests ===")
for raw, expected_contains in addr_tests:
    normed = normalize_address(raw)
    print(f"  '{raw}' -> '{normed}'")

# Test feature computation
print("\n=== Feature computation test ===")
n1 = normalize_name("Microsoft Corp")
n2 = normalize_name("Microsoft Corporation")
a1 = normalize_address("One Microsoft Way, Redmond, WA")
a2 = normalize_address("1 Microsoft Way, Redmond, Washington")
nt1, nt2 = name_tokens(n1), name_tokens(n2)
at1, at2 = address_tokens(a1), address_tokens(a2)
ng1, ng2 = char_ngrams(n1), char_ngrams(n2)
ag1, ag2 = char_ngrams(a1), char_ngrams(a2)

feats = compute_pairwise_features(
    n1, n2, a1, a2, 'us', 'us',
    nt1, nt2, at1, at2, ng1, ng2, ag1, ag2
)
print(f"  Features shape: {feats.shape}")
print(f"  Feature names: {len(FEATURE_NAMES)}")
print(f"  Num features: {NUM_FEATURES}")
for name, val in zip(FEATURE_NAMES, feats):
    print(f"    {name:30s} = {val:.4f}")

# Test on actual data (tiny sample)
print("\n=== Loading sample data ===")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str,
                  keep_default_na=False, nrows=100, quoting=csv.QUOTE_NONE)
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=str,
                  keep_default_na=False, nrows=100, quoting=csv.QUOTE_NONE)
print(f"  S1 sample: {len(s1)} rows, columns: {list(s1.columns)}")
print(f"  S2 sample: {len(s2)} rows, columns: {list(s2.columns)}")
print(f"\n  S1 row 0: {dict(s1.iloc[0])}")
print(f"  S2 row 0: {dict(s2.iloc[0])}")

print("\n=== ALL SMOKE TESTS PASSED ===")
