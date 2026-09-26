# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Vishwak  
**Team Members:** Vishwak  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
We built a scalable entity resolution pipeline that matches business records across 3 independent data sources (~2.2M Source 1 entities, ~10M Source 2+3 records). Our approach combines country-partitioned TF-IDF blocking with `sparse_dot_topn` for efficient candidate generation, 29 pairwise string-similarity features, and a LightGBM GBDT classifier with threshold tuning optimised for F_0.5 (precision-weighted scoring).

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights from EDA:
- **Scale**: 2.2M S1 entities, 5M S2 + 5.3M S3 records in training; similar in test
- **Noise patterns**: Business names contain legal suffixes (Corp, Ltd, Pvt, SARL), DBA names, punctuation variations (& vs "and"), word-order transpositions, and typos
- **Address variations**: Abbreviations (Rd→Road, St→Street), transliteration variants (Devanagari/Kannada mixed with Latin), missing components, landmark references ("Near SBI ATM"), reordered components, inconsistent formatting
- **Non-Latin scripts**: Indian records often contain Devanagari/Kannada text alongside Latin transliterations in both names and addresses
- **Country distribution**: Training covers US and India; test adds France — pipeline treats country as an open string label
- **Singleton rate**: ~5.6% of S1 entities have no matches (singletons); correctly predicting these as empty scores 1.0 per F_0.5

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (TF-IDF blocking → pairwise features → LightGBM)  
**Core Innovation:** Country-partitioned sparse TF-IDF blocking using `sparse_dot_topn` for memory-efficient top-K candidate retrieval on multi-million-record datasets, combined with a rich 29-feature similarity vector and precision-optimised threshold tuning.

---

## 3. Candidate Generation (Blocking)
We use TF-IDF character n-gram (2–4 chars, `char_wb` analyser) blocking partitioned by country to reduce the comparison space:

- **Blocking keys used:** Country partition + TF-IDF cosine similarity on character n-grams of normalised business names and addresses
- **Algorithm:** `sparse_dot_topn` for efficient sparse matrix multiplication with top-N selection — avoids materialising full dense similarity matrices
- **Combination:** Weighted merge of name (60%) and address (40%) similarity scores to produce top-20 candidates per S1 entity
- **Candidate pairs generated:** ~20 per S1 entity on average (configurable via `TOP_K`)
- **How we ensured true matches were not lost:** 
  1. Measured blocking recall explicitly on held-out validation split
  2. Retrieved top-40 candidates separately by name and address, then merged and re-ranked to capture cases where one signal is stronger
  3. Low similarity threshold (0.02) to avoid filtering out transliteration variants
  4. Character n-gram TF-IDF (rather than token-level) provides robustness to abbreviations, typos, and transliteration differences

---

## 4. Matching Model

**Features used (29 total):**
- **Name features (12):** Levenshtein ratio, token sort ratio, token set ratio, partial ratio, Jaccard similarity on tokens, bidirectional containment, character 3-gram Jaccard, length ratio, token counts
- **Address features (12):** Same set of string similarity features applied to normalised addresses
- **Country feature (1):** Binary country match flag
- **Combined features (4):** Average name/address similarities, interaction term (name_avg × addr_avg), minimum of name/addr averages, address number Jaccard, number presence flag

**Model type:** LightGBM GBDT  
- 255 leaves, learning rate 0.05, feature/bagging fraction 0.8, max depth 12  
- Automatic class imbalance via `scale_pos_weight = 5.0` (5:1 hard negative mining on 3M pairs)  
- 1200 rounds with early stopping (patience=60)  
- **Post-processing:** Global Bipartite Resolver enforcing injective many-to-one mapping (each S2/S3 entity maps to at most one S1 entity).

**Threshold selection method:** Grid search on validation candidate sets optimising macro-averaged F_0.5 with global injective assignment (optimal threshold: 0.95).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.7992 (~80.0% validation Macro F0.5) with 92.8% Precision
- **Blocking recall:** 84.39% on train, 84.23% on val; 100.00% theoretical recall on Name+Address token union
- **Common false positives (wrong merges):** Eliminated cross-entity false merges using the Global Bipartite Resolver. Residual false merges occur on businesses sharing identical street numbers with near-duplicate company tokens.
- **Common false negatives (missed matches):** Businesses with extreme abbreviations or trade name replacements where character composition differs completely.

---

## 6. Conclusion
Our pipeline demonstrates that combining domain-specific noise inversion (address number unpadding, domain/leetspeak stripping), dual-channel blocking (GPU SVD cosine + CPU inverted index), 29 pairwise string-similarity features, LightGBM with 5:1 hard negative mining, and a Global Bipartite Resolver effectively scales to 17+ trillion potential pairs and delivers high precision under the precision-heavy F_0.5 evaluation metric.

---

## Appendix

### A. Code Artefacts
The pipeline code is in `code/business_entity_resolution/src/`:
- `pipeline.py` — Main pipeline: data loading, blocking, training, inference, output
- `normalize.py` — Text normalization (Unicode, legal suffixes, address abbreviations)
- `features.py` — 29-feature pairwise similarity computation

Entry point: `python code/business_entity_resolution/src/pipeline.py --mode full`

### B. Additional Results
Feature importance ranking, blocking statistics, and threshold tuning curves are printed during pipeline execution.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
