import sys, time
sys.path.insert(0, 'code/business_entity_resolution/src')
from normalize import normalize_name, normalize_address

tests = [
    ('Microsoft Corp.', 'microsoft'),
    ('Tata Consultancy Services Private Limited', 'tata consultancy services'),
    ('SOCIETE GENERALE SA', 'societe generale sa'),
    ("Ben & Jerry's Ice Cream LLC", 'ben and jerry ice cream'),
]
for raw, exp in tests:
    got = normalize_name(raw)
    ok = exp in got
    print(f"  {raw} -> {got}  [{'OK' if ok else 'FAIL'}]")

import pandas as pd, csv
df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str,
                  keep_default_na=False, nrows=50000, quoting=csv.QUOTE_NONE)
t0 = time.time()
df['name_norm'] = df['business_name'].apply(normalize_name)
df['addr_norm'] = df['business_address'].apply(normalize_address)
elapsed = time.time() - t0
print(f"\nNormalized 50k rows in {elapsed:.2f}s  ({50000/elapsed:.0f} rows/s)")
print(f"Estimated time for 12M rows: {12e6/50000*elapsed/60:.1f} min")
