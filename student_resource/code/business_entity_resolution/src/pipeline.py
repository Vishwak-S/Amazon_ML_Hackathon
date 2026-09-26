"""
pipeline.py — MAX ACCURACY Entity Resolution Pipeline.

Hardware: 32-core CPU + RTX 4070 Laptop (8 GB VRAM) + 16 GB RAM

DUAL BLOCKING ARCHITECTURE (biggest accuracy lever):
  1. TF-IDF char-(2,4)-gram + SVD(64D) cosine → TOP_K=50 per entity  (GPU)
  2. Inverted token index on significant name tokens → TOP_K=50 per entity  (CPU)
  3. UNION of both → blocking recall ~82-90%  (was 56% with cosine/K=10 alone)

WHY DUAL BLOCKING MATTERS:
  - Cosine fails when names differ in word order / abbreviation / transliteration
  - Token index catches these cases if they share even one rare word
  - Union gives both precision AND recall

Expected score: ~80-88% Macro F0.5  (was 48% with broken neg-sampling)
For 98%+: would need pre-trained multilingual sentence transformers (SBERT etc.)
"""

import argparse, csv, gc, os, pickle, sys, time
import multiprocessing

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (normalize_name, normalize_address,
                       name_tokens, address_tokens, char_ngrams)
from features import compute_pairwise_features, FEATURE_NAMES, NUM_FEATURES

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TRAIN_DIR = os.path.join(BASE, 'dataset', 'train')
TEST_DIR  = os.path.join(BASE, 'dataset', 'test')
OUT_DIR   = os.path.join(BASE, 'output')
MDL_DIR   = os.path.join(BASE, 'code', 'business_entity_resolution', 'models')
CACHE_DIR = os.path.join(BASE, 'cache')

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_JOBS = multiprocessing.cpu_count()   # 32 cores
NAME_W = 0.65
ADDR_W = 0.35

TOP_K_COSINE = 60          # cosine candidates per S1 entity
TOP_K_TOKEN  = 40          # token-index candidates per S1 entity
SVD_DIM      = 64          # 64D: richer than 32D; better cosine discrimination
TFIDF_FEATS  = 100_000     # large vocab: better char n-gram coverage
TFIDF_SAMPLE = 300_000
NGRAM_RANGE  = (2, 4)      # bigrams → 4-grams: captures longer patterns

GPU_CHUNK = 128            # 128 × 6.1M × fp16 ≈ 1.6 GB VRAM — safe for RTX 4070
CPU_CHUNK = 100

VAL_FRAC  = 0.15
MAX_POS   = 500_000        # subsample positives so negatives always get a budget
MAX_TRAIN = 3_000_000      # 3M pairs: trains in ~2.5 min with 32 threads
MAX_VAL   = 600_000
NEG_RATIO = 5              # 5 hard negatives per positive

# Token blocking ── only keep tokens that are rare enough to be discriminating
MAX_TOKEN_BUCKET = 2000    # ignore tokens in > 2000 S23 entities (too common)
TOKEN_MIN_LEN    = 4       # ignore tokens shorter than 4 characters

COMPANY_STOPWORDS = {
    # Generic company type words
    'ltd', 'pvt', 'limited', 'inc', 'corp', 'llc', 'co', 'company',
    'private', 'public', 'enterprises', 'enterprise', 'international',
    'services', 'service', 'technologies', 'technology', 'solutions',
    'solution', 'group', 'holdings', 'holding', 'global', 'industries',
    'industry', 'trading', 'associates', 'brothers', 'sons', 'management',
    'systems', 'system', 'engineering', 'construction', 'development',
    'business', 'general', 'national', 'state', 'bank', 'trust', 'fund',
    'care', 'health', 'centre', 'center', 'works',
    # Country/region words
    'india', 'indian', 'us', 'usa', 'france', 'french',
    # European legal forms
    'sa', 'sas', 'gmbh', 'bv', 'nv', 'sarl', 'ste', 'srl', 'ag', 'plc',
    'pte', 'pty', 'aps', 'spa', 'ooo', 'zao',
    # Common English articles / prepositions
    'the', 'and', 'for', 'with',
}

LGB_P = dict(
    objective='binary', metric='binary_logloss', boosting_type='gbdt',
    num_leaves=255, max_depth=12, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=30, lambda_l1=0.05, lambda_l2=0.05,
    verbose=-1, num_threads=N_JOBS,
)
N_BOOST = 1200
EARLY   = 60

gpu_name = torch.cuda.get_device_name(0) if HAS_CUDA else 'None (CPU fallback)'
print(f"[init] {N_JOBS} cores  |  GPU: {gpu_name}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _norm_chunk(chunk):
    return pd.DataFrame({
        'entity_id':    chunk['entity_id'].values,
        'name_norm':    chunk['business_name'].apply(normalize_name).values,
        'addr_norm':    chunk['business_address'].apply(normalize_address).values,
        'country_norm': chunk['country'].str.strip().str.lower().values,
    })

def load_source(path):
    print(f"  Loading {os.path.basename(path)} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    print(f"    {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
    t0  = time.time()
    step    = max(1, len(df) // N_JOBS)
    chunks  = [df.iloc[i:i+step].reset_index(drop=True) for i in range(0, len(df), step)]
    results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_norm_chunk)(c) for c in chunks if len(c) > 0
    )
    out = pd.concat(results, ignore_index=True)
    print(f"    normalised in {time.time()-t0:.1f}s  ({N_JOBS} workers)", flush=True)
    return out

def load_gt(path):
    gt = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f, delimiter='\t'):
            if row[0] == 'source1_entity_id': continue
            gt[row[0]] = set(row[1].split(',')) if len(row) > 1 and row[1].strip() else set()
    wm = sum(1 for v in gt.values() if v)
    print(f"  GT: {len(gt):,} entities, {wm:,} with matches", flush=True)
    return gt

def build_store(df):
    return dict(zip(df['entity_id'], zip(df['name_norm'], df['addr_norm'], df['country_norm'])))


# ══════════════════════════════════════════════════════════════════════════════
# 2a. COSINE BLOCKING  (TF-IDF char n-gram + SVD(64D) + GPU top-K)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_tfidf(name_texts, addr_texts):
    def fit_one(texts):
        s = texts.sample(TFIDF_SAMPLE, random_state=42) if len(texts) > TFIDF_SAMPLE else texts
        return TfidfVectorizer(
            analyzer="char_wb", ngram_range=NGRAM_RANGE,
            max_features=TFIDF_FEATS, sublinear_tf=True, min_df=2,
        ).fit(s)
    return fit_one(name_texts), fit_one(addr_texts)

def cosine_blocking_country(s1c, s23c, top_k):
    print(f"    [Cosine/Sparse] Fitting TF-IDF (vocab={TFIDF_FEATS:,}, ngram={NGRAM_RANGE})...", flush=True)
    import time; t0 = time.time()
    nvec, avec = _fit_tfidf(
        pd.concat([s1c["name_norm"], s23c["name_norm"]]),
        pd.concat([s1c["addr_norm"],  s23c["addr_norm"]]),
    )
    print(f"    TF-IDF fit: {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    s23n = nvec.transform(s23c["name_norm"])
    s23a = avec.transform(s23c["addr_norm"])
    s23_emb = sp.hstack([s23n * NAME_W, s23a * ADDR_W]).tocsr()
    normalize(s23_emb, norm="l2", copy=False)
    del s23n, s23a; import gc; gc.collect()
    print(f"    S23 embedded: {time.time()-t0:.0f}s  shape={s23_emb.shape}", flush=True)

    t0 = time.time()
    s1n = nvec.transform(s1c["name_norm"])
    s1a = avec.transform(s1c["addr_norm"])
    s1_emb = sp.hstack([s1n * NAME_W, s1a * ADDR_W]).tocsr()
    normalize(s1_emb, norm="l2", copy=False)
    del s1n, s1a, nvec, avec; gc.collect()
    print(f"    S1  embedded: {time.time()-t0:.0f}s  shape={s1_emb.shape}", flush=True)

    t0 = time.time()
    # sparse_dot_topn (multi-threaded, no CUDA needed)
    C = sp_matmul_topn(s1_emb, s23_emb.T, top_n=top_k, n_threads=N_JOBS)
    del s1_emb, s23_emb; gc.collect()
    
    # Extract candidates
    s23_ids = s23c["entity_id"].values
    s1_ids = s1c["entity_id"].values
    res = {}
    
    import numpy as np
    # C is CSR. We can iterate over rows.
    for i in range(C.shape[0]):
        start = C.indptr[i]
        end = C.indptr[i+1]
        indices = C.indices[start:end]
        # sort by score descending (sp_matmul_topn doesn't strictly sort)
        scores = C.data[start:end]
        if len(scores) > 0:
            order = np.argsort(-scores)
            res[s1_ids[i]] = s23_ids[indices[order]].tolist()
        else:
            res[s1_ids[i]] = []
            
    print(f"    Top-{top_k} search: {time.time()-t0:.0f}s", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 2b. TOKEN BLOCKING  (inverted index on significant name tokens)
# ══════════════════════════════════════════════════════════════════════════════

ADDR_STOPWORDS = {
    'road', 'street', 'avenue', 'lane', 'drive', 'rd', 'st', 'ave', 'dr',
    'suite', 'floor', 'near', 'opp', 'dept', 'boulevard', 'blvd', 'highway',
    'hwy', 'circle', 'cir', 'court', 'ct', 'place', 'pl', 'square', 'sq',
    'unit', 'building', 'bldg', 'room', 'rm', 'box', 'pob', 'post', 'office',
    'north', 'south', 'east', 'west', 'city', 'state', 'town',
}

def _sig_tokens(name, addr):
    """Extract discriminating tokens from both name and address."""
    toks = []
    for t in name.split():
        if len(t) >= 3 and t not in COMPANY_STOPWORDS:
            toks.append(t)
    for w in addr.split():
        if w.isdigit() and len(w) >= 2:
            toks.append(w)
        elif len(w) >= 4 and w not in ADDR_STOPWORDS and w not in COMPANY_STOPWORDS:
            toks.append(w)
    return set(toks)

def token_blocking_country(s1c, s23c, top_k):
    """
    Build inverted index on S23 name + address tokens.
    For each S1 entity, retrieve top-K S23 entities by shared-token count.
    """
    print(f"    [Token] Building inverted index ({len(s23c):,} S23 entities)...", flush=True)
    t0 = time.time()

    inv_idx = {}
    for eid, name, addr in zip(s23c['entity_id'].values, s23c['name_norm'].values, s23c['addr_norm'].values):
        for tok in _sig_tokens(name, addr):
            if tok in inv_idx:
                inv_idx[tok].append(eid)
            else:
                inv_idx[tok] = [eid]

    # Filter: drop tokens in > 1000 entities (too generic) or only 1 entity
    inv_idx = {t: ids for t, ids in inv_idx.items() if 1 < len(ids) <= 1000}
    print(f"    Token index built: {len(inv_idx):,} discriminating tokens in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    result = {}
    s1_eids = s1c['entity_id'].values
    s1_nms  = s1c['name_norm'].values
    s1_ads  = s1c['addr_norm'].values

    for eid, nm, ad in zip(tqdm(s1_eids, desc="    Token query", mininterval=4), s1_nms, s1_ads):
        counter = {}
        for tok in _sig_tokens(nm, ad):
            postings = inv_idx.get(tok)
            if postings:
                for cid in postings:
                    counter[cid] = counter.get(cid, 0) + 1
        if counter:
            result[eid] = sorted(counter, key=counter.get, reverse=True)[:top_k]
        else:
            result[eid] = []

    print(f"    Token lookup done: {time.time()-t0:.0f}s", flush=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2c. COMBINED BLOCKING (union of cosine + token, with checkpointing)
# ══════════════════════════════════════════════════════════════════════════════

def blocking(s1, s23, split_tag='train'):
    print(f"\n== BLOCKING [{split_tag.upper()}] (Cosine+Token union) ==", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cands     = {}
    countries = sorted(set(s1['country_norm'].unique()) | set(s23['country_norm'].unique()))
    print(f"  Countries: {countries}", flush=True)

    for ctry in countries:
        cpath = os.path.join(CACHE_DIR, f"blk_{split_tag}_{ctry}.pkl")
        if os.path.exists(cpath):
            print(f"\n  [{ctry}] Loading cached checkpoint...", flush=True)
            with open(cpath, 'rb') as f:
                ctry_cands = pickle.load(f)
            cands.update(ctry_cands)
            tot = sum(len(v) for v in ctry_cands.values())
            print(f"    Loaded {len(ctry_cands):,} entities  avg {tot/max(len(ctry_cands),1):.1f}/entity", flush=True)
            continue

        s1c  = s1 [s1 ['country_norm'] == ctry].reset_index(drop=True)
        s23c = s23[s23['country_norm'] == ctry].reset_index(drop=True)

        if len(s1c) == 0: continue
        if len(s23c) == 0:
            ctry_cands = {sid: [] for sid in s1c['entity_id']}
            cands.update(ctry_cands)
            continue

        print(f"\n  [{ctry}] S1={len(s1c):,}  S23={len(s23c):,}", flush=True)

        # Cosine blocking (GPU)
        cosine_cands = cosine_blocking_country(s1c, s23c, TOP_K_COSINE)

        # Token blocking (CPU — fast inverted index)
        token_cands  = token_blocking_country(s1c, s23c, TOP_K_TOKEN)

        # Union
        ctry_cands = {}
        for eid in s1c['entity_id'].values:
            cos_list = cosine_cands.get(eid, [])
            tok_list = token_cands.get(eid, [])
            # Deduplicate while preserving cosine-first order (higher quality ranking first)
            seen     = set()
            merged   = []
            for cid in cos_list + tok_list:
                if cid not in seen:
                    seen.add(cid)
                    merged.append(cid)
            ctry_cands[eid] = merged

        n_total = sum(len(v) for v in ctry_cands.values())
        print(f"    Union: avg {n_total/max(len(ctry_cands),1):.1f} candidates/entity"
              f"  (cosine-only avg: {TOP_K_COSINE}, token adds ~{n_total/max(len(ctry_cands),1)-TOP_K_COSINE:.0f} more)", flush=True)
        cands.update(ctry_cands)

        with open(cpath, 'wb') as f:
            pickle.dump(ctry_cands, f)
        print(f"    Checkpoint saved: {cpath}", flush=True)

    for sid in s1['entity_id']:
        cands.setdefault(sid, [])

    tot = sum(len(v) for v in cands.values())
    print(f"\n  Total candidates: {tot:,}  avg {tot/max(len(cands),1):.1f}/entity", flush=True)
    return cands

def blk_recall(cands, gt):
    total = found = 0
    for sid, trues in gt.items():
        if not trues: continue
        total += len(trues)
        found += len(trues & set(cands.get(sid, [])))
    r = found / total if total else 1.
    print(f"  Blocking recall: {found:,}/{total:,} = {r:.4f}", flush=True)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 3. PARALLEL PAIRWISE FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _feats_chunk(pairs_chunk, s1_store, s23_store):
    n = len(pairs_chunk)
    X = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        a, b = pairs_chunk[i]
        n1, a1, c1 = s1_store[a]
        n2, a2, c2 = s23_store[b]
        X[i] = compute_pairwise_features(
            n1, n2, a1, a2, c1, c2,
            name_tokens(n1), name_tokens(n2),
            address_tokens(a1), address_tokens(a2),
            char_ngrams(n1, 3), char_ngrams(n2, 3),
            char_ngrams(a1, 3), char_ngrams(a2, 3),
        )
    return X

def feats_batch(pairs, s1_store, s23_store, n_jobs=N_JOBS):
    """
    Parallel feature extraction using joblib loky.
    Key trick: pass only needed store entries per batch → minimizes IPC serialization.
    """
    if not pairs:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)

    # Build minimal sub-stores for this batch
    need_s1  = {a for a, b in pairs}
    need_s23 = {b for a, b in pairs}
    mini_s1  = {k: s1_store[k]  for k in need_s1  if k in s1_store}
    mini_s23 = {k: s23_store[k] for k in need_s23 if k in s23_store}

    # Cap workers at 16 and chunk size at 20,000 for rock-solid RAM stability
    n_jobs_actual = min(16, n_jobs, max(1, len(pairs) // 20_000))
    if n_jobs_actual <= 1:
        return _feats_chunk(pairs, mini_s1, mini_s23)

    sz     = max(10_000, len(pairs) // n_jobs_actual)
    chunks = [pairs[i:i+sz] for i in range(0, len(pairs), sz)]
    parts  = Parallel(n_jobs=n_jobs_actual, backend='loky')(
        delayed(_feats_chunk)(ch, mini_s1, mini_s23) for ch in chunks
    )
    return np.vstack(parts).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PAIR GENERATION (hard negatives)
# ══════════════════════════════════════════════════════════════════════════════

def build_pairs(cands, gt, s23_ids_set,
                neg_ratio=NEG_RATIO, max_p=MAX_TRAIN, max_pos=MAX_POS):
    rng  = np.random.RandomState(42)
    pos, neg = [], []

    for sid, trues in tqdm(gt.items(), desc="  pairs", mininterval=2):
        cs = set(cands.get(sid, []))
        if not cs: continue
        p  = [m for m in (trues & cs) if m in s23_ids_set]
        n  = [m for m in (cs - trues) if m in s23_ids_set]
        pos.extend((sid, m) for m in p)
        if n:
            k   = min(len(n), max(neg_ratio * len(p), 1))
            sel = rng.choice(n, size=k, replace=False) if k < len(n) else n
            neg.extend((sid, m) for m in sel)

    # Cap positives first so negatives always fit in budget
    if len(pos) > max_pos:
        idx = rng.choice(len(pos), size=max_pos, replace=False)
        pos = [pos[i] for i in idx]

    max_neg = max_p - len(pos)
    if max_neg > 0 and len(neg) > max_neg:
        idx = rng.choice(len(neg), size=max_neg, replace=False)
        neg = [neg[i] for i in idx]
    elif max_neg <= 0:
        neg = []

    print(f"  Pairs: pos={len(pos):,}  neg={len(neg):,}  ratio={len(neg)/max(len(pos),1):.1f}:1", flush=True)
    all_p = pos + neg
    y     = np.array([1]*len(pos) + [0]*len(neg), dtype=np.float32)
    perm  = rng.permutation(len(all_p))
    return [all_p[i] for i in perm], y[perm]


# ══════════════════════════════════════════════════════════════════════════════
# 5. LIGHTGBM TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_lgb(Xtr, ytr, Xva, yva):
    print("\n== TRAINING ==", flush=True)
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, feature_name=FEATURE_NAMES,
                      free_raw_data=False, reference=dtr)
    p   = LGB_P.copy()
    n_neg = len(ytr) - ytr.sum()
    p['scale_pos_weight'] = max(0.1, n_neg / max(ytr.sum(), 1))
    print(f"  scale_pos_weight={p['scale_pos_weight']:.2f}  threads={p['num_threads']}", flush=True)
    model = lgb.train(p, dtr, num_boost_round=N_BOOST,
                      valid_sets=[dtr, dva], valid_names=['tr', 'va'],
                      callbacks=[lgb.log_evaluation(100), lgb.early_stopping(EARLY)])
    print(f"  best iter: {model.best_iteration}", flush=True)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 6. THRESHOLD TUNING (Macro F_0.5)
# ══════════════════════════════════════════════════════════════════════════════

def f05_macro(preds, gt):
    scores = []
    for sid in gt:
        t = gt[sid]; p = preds.get(sid, set())
        if not t and not p: scores.append(1.)
        elif not t or not p: scores.append(0.)
        else:
            tp = len(t & p)
            pr = tp/len(p); rc = tp/len(t)
            scores.append((1.25*pr*rc)/(0.25*pr+rc) if pr+rc else 0.)
    return float(np.mean(scores)) if scores else 0.

def tune_thr(model, vpairs, Xva, vgt, s1_store, s23_store):
    print("\n== THRESHOLD OPTIMIZATION (Global Bipartite Many-to-One) ==", flush=True)
    probs = model.predict(Xva, num_iteration=model.best_iteration)

    evaluated_sids = {a for a, b in vpairs}
    eval_gt = {sid: vgt[sid] for sid in evaluated_sids if sid in vgt}

    best, best_t = 0., 0.85
    for t in np.arange(0.60, 0.98, 0.02):
        best_assign = {}
        for (a, b), p in zip(vpairs, probs):
            # EXACT FAST PATH
            n1, a1, c1 = s1_store[a]
            n2, a2, c2 = s23_store[b]
            if n1 and n1 == n2 and a1 and a1 == a2:
                p = 1.0
                
            if p >= t:
                curr = best_assign.get(b)
                if curr is None or p > curr[1]:
                    best_assign[b] = (a, p)
        preds = {sid: set() for sid in eval_gt}
        for b, (a, p) in best_assign.items():
            if a in preds:
                preds[a].add(b)
        sc = f05_macro(preds, eval_gt)
        if sc > best: best, best_t = sc, t

    for t in np.arange(max(.50, best_t-.04), min(.99, best_t+.04), .005):
        best_assign = {}
        for (a, b), p in zip(vpairs, probs):
            # EXACT FAST PATH
            n1, a1, c1 = s1_store[a]
            n2, a2, c2 = s23_store[b]
            if n1 and n1 == n2 and a1 and a1 == a2:
                p = 1.0
                
            if p >= t:
                curr = best_assign.get(b)
                if curr is None or p > curr[1]:
                    best_assign[b] = (a, p)
        preds = {sid: set() for sid in eval_gt}
        for b, (a, p) in best_assign.items():
            if a in preds:
                preds[a].add(b)
        sc = f05_macro(preds, eval_gt)
        if sc > best: best, best_t = sc, t

    print(f"  Optimal thr={best_t:.4f}  Validation F0.5={best:.4f}", flush=True)
    return best_t, best


# ══════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE (streaming, parallel features, Bipartite Injective Matching)
# ══════════════════════════════════════════════════════════════════════════════

def inference(model, cands, s1_store, s23_store, thr, all_ids):
    print(f"\n== INFERENCE (thr={thr:.4f}) with Global Bipartite Resolver ==", flush=True)
    cout     = {}
    BSIZE    = 500_000
    batch    = []
    total    = 0
    pbar     = tqdm(total=len(all_ids), desc="  scoring", mininterval=5)

    # Injective tracking: cid -> (best_sid, max_prob)
    # A single S2/S3 entity can ONLY belong to at most ONE Source 1 entity!
    best_assignment = {}

    for sid in all_ids:
        cs = [c for c in cands.get(sid, []) if c in s23_store]
        cout[sid] = cs
        
        # EXACT MATCH FAST PATH
        n1, a1, c1 = s1_store[sid]
        exact_matches = []
        for c in cs:
            n2, a2, c2 = s23_store[c]
            if n1 and n1 == n2 and a1 and a1 == a2:
                exact_matches.append(c)
                
        for c in exact_matches:
            curr = best_assignment.get(c)
            if curr is None or 1.0 > curr[1]:
                best_assignment[c] = (sid, 1.0)
                
        # Only evaluate non-exact matches through LightGBM
        cs = [c for c in cs if c not in exact_matches]
        
        batch.extend((sid, c) for c in cs)
        total += len(cs)

        if len(batch) >= BSIZE:
            X     = feats_batch(batch, s1_store, s23_store)
            probs = model.predict(X, num_iteration=model.best_iteration)
            for (s, cid), p in zip(batch, probs):
                if p >= thr:
                    curr = best_assignment.get(cid)
                    if curr is None or p > curr[1]:
                        best_assignment[cid] = (s, p)
            batch = []; del X, probs; gc.collect()
        pbar.update(1)

    if batch:
        X     = feats_batch(batch, s1_store, s23_store)
        probs = model.predict(X, num_iteration=model.best_iteration)
        for (s, cid), p in zip(batch, probs):
            if p >= thr:
                curr = best_assignment.get(cid)
                if curr is None or p > curr[1]:
                    best_assignment[cid] = (s, p)
        del batch, X, probs; gc.collect()

    pbar.close()
    print(f"  Scored {total:,} candidate pairs.", flush=True)

    # Assemble final matching dictionary
    matching = {sid: [] for sid in all_ids}
    for cid, (s, p) in best_assignment.items():
        matching[s].append(cid)

    for sid in matching:
        matching[sid].sort()

    return matching, cout


# ══════════════════════════════════════════════════════════════════════════════
# 8. OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def write_out(matching, cout, all_ids):
    os.makedirs(OUT_DIR, exist_ok=True)
    mp = os.path.join(OUT_DIR, 'matching_results.tsv')
    cp = os.path.join(OUT_DIR, 'candidate_pairs.tsv')

    with open(mp, 'w', encoding='utf-8', newline='') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for sid in all_ids:
            ids = [m for m in dict.fromkeys(matching.get(sid, [])) if m.startswith(('S2-','S3-'))]
            f.write(f"{sid}\t{','.join(ids)}\n")

    with open(cp, 'w', encoding='utf-8', newline='') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for sid in all_ids:
            cs    = cout.get(sid, [])
            ms    = set(matching.get(sid, []))
            all_c = [c for c in dict.fromkeys(cs + [m for m in ms if m not in set(cs)])
                     if c.startswith(('S2-','S3-'))]
            f.write(f"{sid}\t{','.join(all_c)}\n")

    print(f"  Wrote: {mp}", flush=True)
    print(f"  Wrote: {cp}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_train():
    t0 = time.time()
    print("="*70+"\n ENTITY RESOLUTION — TRAIN\n"+"="*70, flush=True)

    print("\n-- LOAD --", flush=True)
    s1  = load_source(os.path.join(TRAIN_DIR, 'train_source1.tsv'))
    s2  = load_source(os.path.join(TRAIN_DIR, 'train_source2.tsv'))
    s3  = load_source(os.path.join(TRAIN_DIR, 'train_source3.tsv'))
    gt  = load_gt(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'))
    s23 = pd.concat([s2, s3], ignore_index=True); del s2, s3; gc.collect()
    print(f"  S23: {len(s23):,}", flush=True)

    print("\n-- VAL SPLIT --", flush=True)
    rng = np.random.RandomState(42)
    ids = list(gt.keys()); rng.shuffle(ids)
    vn  = int(len(ids) * VAL_FRAC)
    vid, tid = set(ids[:vn]), set(ids[vn:])
    tgt = {k: v for k, v in gt.items() if k in tid}
    vgt = {k: v for k, v in gt.items() if k in vid}
    s1t = s1[s1['entity_id'].isin(tid)].reset_index(drop=True)
    s1v = s1[s1['entity_id'].isin(vid)].reset_index(drop=True)
    print(f"  Train: {len(tid):,}   Val: {len(vid):,}", flush=True)

    print("\n-- BUILD STORES --", flush=True)
    s1_store    = build_store(s1)
    s23_store   = build_store(s23)
    s23_ids_set = set(s23['entity_id'])

    print("\n-- TRAIN BLOCKING --", flush=True)
    tc = blocking(s1t, s23, split_tag='train'); blk_recall(tc, tgt); del s1t; gc.collect()

    print("\n-- VAL BLOCKING --", flush=True)
    vc = blocking(s1v, s23, split_tag='val'); blk_recall(vc, vgt); del s1v, s23; gc.collect()

    print("\n-- PAIRS & FEATURES --", flush=True)
    tp, yt = build_pairs(tc, tgt, s23_ids_set, neg_ratio=NEG_RATIO, max_p=MAX_TRAIN, max_pos=MAX_POS)
    del tc; gc.collect()
    print("  Computing train features ...", flush=True)
    t1  = time.time()
    Xtr = feats_batch(tp, s1_store, s23_store); del tp; gc.collect()
    print(f"  Train features done: {time.time()-t1:.0f}s", flush=True)

    vp, yv = build_pairs(vc, vgt, s23_ids_set, neg_ratio=NEG_RATIO, max_p=MAX_VAL, max_pos=MAX_POS//2)
    del vc, s23_ids_set; gc.collect()
    print("  Computing val features ...", flush=True)
    t1  = time.time()
    Xva = feats_batch(vp, s1_store, s23_store)
    print(f"  Val features done: {time.time()-t1:.0f}s", flush=True)

    model    = train_lgb(Xtr, yt, Xva, yv); del Xtr, yt; gc.collect()
    thr, f05 = tune_thr(model, vp, Xva, vgt, s1_store, s23_store)
    del vp, Xva, s1_store, s23_store; gc.collect()

    os.makedirs(MDL_DIR, exist_ok=True)
    model.save_model(os.path.join(MDL_DIR, 'lgb.txt'))
    with open(os.path.join(MDL_DIR, 'cfg.pkl'), 'wb') as f:
        pickle.dump({'thr': thr, 'f05': f05}, f)

    imp = model.feature_importance(importance_type='gain')
    print("\n  Top-10 features:", flush=True)
    for i in np.argsort(imp)[::-1][:10]:
        print(f"    {FEATURE_NAMES[i]:30s} {imp[i]:10.1f}", flush=True)

    print(f"\n  Training complete: {(time.time()-t0)/60:.1f} min", flush=True)
    return model, thr, f05

def run_test(model=None, thr=None):
    t0 = time.time()
    print("\n"+"="*70+"\n ENTITY RESOLUTION — TEST\n"+"="*70, flush=True)
    if model is None:
        model = lgb.Booster(model_file=os.path.join(MDL_DIR, 'lgb.txt'))
    if thr is None:
        with open(os.path.join(MDL_DIR, 'cfg.pkl'), 'rb') as f:
            thr = pickle.load(f)['thr']
    print(f"  Threshold: {thr:.4f}", flush=True)

    s1  = load_source(os.path.join(TEST_DIR, 'test_source1.tsv'))
    s2  = load_source(os.path.join(TEST_DIR, 'test_source2.tsv'))
    s3  = load_source(os.path.join(TEST_DIR, 'test_source3.tsv'))
    s23 = pd.concat([s2, s3], ignore_index=True); del s2, s3; gc.collect()

    s1_store  = build_store(s1)
    s23_store = build_store(s23)
    cands     = blocking(s1, s23, split_tag='test'); del s23; gc.collect()
    all_ids   = s1['entity_id'].tolist(); del s1; gc.collect()

    matching, cout = inference(model, cands, s1_store, s23_store, thr, all_ids)
    del s1_store, s23_store; gc.collect()
    write_out(matching, cout, all_ids)

    ns = sum(1 for v in matching.values() if not v)
    nm = sum(1 for v in matching.values() if v)
    print(f"\n  singletons={ns:,}  matched={nm:,}  done in {(time.time()-t0)/60:.1f} min", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'test', 'full'], default='full')
    a  = ap.parse_args()
    if   a.mode == 'train': run_train()
    elif a.mode == 'test':  run_test()
    else:
        m, t, _ = run_train()
        run_test(m, t)

if __name__ == '__main__':
    main()
