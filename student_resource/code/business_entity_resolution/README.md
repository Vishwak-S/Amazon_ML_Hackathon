# Business Entity Resolution Pipeline

## Overview

An end-to-end ML pipeline that matches business records across 3 independent data sources using TF-IDF blocking, pairwise string-similarity features, and LightGBM classification.

## Prerequisites

- Python 3.10+
- ~32 GB RAM (for processing the full dataset)
- GPU optional (LightGBM can use CPU; the pip wheel doesn't include GPU support by default)

## Setup

```bash
pip install -r requirements.txt
```

## Data Layout

Place the dataset under `../../dataset/` relative to this directory:

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── pipeline.py      # Main pipeline (train + inference)
│   │   ├── normalize.py     # Text normalization
│   │   └── features.py      # Pairwise feature engineering
│   ├── requirements.txt
│   └── README.md
└── output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

## Reproduction Steps

### 1. Full Pipeline (Train + Inference)

Run from the `student_resource/` directory:

```bash
cd student_resource
python code/business_entity_resolution/src/pipeline.py --mode full
```

This will:
1. Load and normalize all training/test data
2. Hold out 15% of training entities for validation
3. Build TF-IDF blocking per country using sparse_dot_topn
4. Generate pairwise features (29 string similarity features)
5. Train a LightGBM classifier
6. Tune the classification threshold to maximize F_0.5
7. Run inference on the test set
8. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`

### 2. Train Only

```bash
python code/business_entity_resolution/src/pipeline.py --mode train
```

Saves the model to `code/business_entity_resolution/models/`.

### 3. Test Only (requires saved model)

```bash
python code/business_entity_resolution/src/pipeline.py --mode test
```

### 4. Validate Output

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Architecture

### Normalization (`normalize.py`)
- Unicode NFKD decomposition → ASCII transliteration
- Strip non-Latin scripts (Devanagari, etc.)
- Lowercase, remove punctuation
- Standardize legal suffixes (Corp→∅, Pvt Ltd→∅, etc.)
- Expand address abbreviations (Rd→Road, St→Street, etc.)

### Blocking (`pipeline.py::blocking()`)
- Country-partitioned to reduce comparison space
- TF-IDF char n-grams (2–4) on names and addresses
- `sparse_dot_topn` for efficient top-K sparse cosine similarity
- Combined score: 0.6 × name_sim + 0.4 × addr_sim
- Top-20 candidates per S1 entity

### Features (`features.py`)
29 pairwise features including:
- **Name**: Levenshtein ratio, token sort/set ratio, partial ratio, Jaccard, containment, char n-gram Jaccard, length ratio
- **Address**: Same set of string similarity features
- **Combined**: avg similarities, interaction, min, number overlap, country match

### Model
- LightGBM GBDT classifier
- Automatic class imbalance handling via `scale_pos_weight`
- 2000 rounds with early stopping (100 rounds patience)

### Threshold Tuning
- Grid search over thresholds on validation split
- Optimizes F_0.5 (precision-weighted 2× over recall)
- Fine-grained search around best coarse threshold

## License

All dependencies are MIT/Apache-2.0 licensed. LightGBM is MIT licensed.
