#!/usr/bin/env python
# coding: utf-8

# # Part B Completion: Export Top-100 High-Risk (Uncertain) Examples
#
# Prerequisites (already produced by Part A / Part B uncertainty step):
#   - patents_50k_green.parquet
#   - logistic_regression_classifier.joblib
#   - X_train_emb.npy, X_eval_emb.npy  (not needed here)
#   - prob_train.npy, prob_eval.npy     (not needed here)
#
# This script:
#   1. Loads pool_unlabeled split
#   2. Encodes with frozen PatentSBERTa
#   3. Predicts p_green with the saved LR classifier
#   4. Computes u = 1 - 2 * |p - 0.5|
#   5. Selects top-100 highest-u rows
#   6. Exports hitl_green_100.csv

import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "AI-Growth-Lab/PatentSBERTa"
ROOT = Path(__file__).resolve().parent.parent

PARQUET_PATH = ROOT / "patents_50k_green.parquet"
CLF_PATH     = ROOT / "logistic_regression_classifier.joblib"
OUT_CSV      = ROOT / "hitl_green_100.csv"
TOP_K        = 100
BATCH_SIZE   = 64

# ── 1. Load pool_unlabeled ──────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
pool = df[df["split"] == "pool_unlabeled"].copy().reset_index(drop=True)
print(f"pool_unlabeled size: {len(pool)}")

# Add a stable doc_id so the human reviewer can reference rows unambiguously
pool["doc_id"] = pool.index.map(lambda i: f"pool_{i:05d}")

# ── 2. Encode with frozen PatentSBERTa ─────────────────────────────────────────
enc = SentenceTransformer(MODEL_NAME, device="cuda")
X_pool = enc.encode(
    pool["text"].astype(str).tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print(f"Encoded shape: {X_pool.shape}")

# ── 3. Load classifier & predict p_green ───────────────────────────────────────
clf = joblib.load(CLF_PATH)
p_green = clf.predict_proba(X_pool)[:, 1]   # probability of positive class

# ── 4. Compute uncertainty score ───────────────────────────────────────────────
u = 1.0 - 2.0 * np.abs(p_green - 0.5)

pool["p_green"] = p_green.astype(np.float32)
pool["u"]       = u.astype(np.float32)

# ── 5. Select top-100 most uncertain ───────────────────────────────────────────
top100 = (
    pool.nlargest(TOP_K, "u")
        .reset_index(drop=True)
)
print(f"\nTop-{TOP_K} uncertainty range:  u_min={top100['u'].min():.4f}  u_max={top100['u'].max():.4f}")
print(f"p_green range in top-{TOP_K}:    p_min={top100['p_green'].min():.4f}  p_max={top100['p_green'].max():.4f}")

# ── 6. Export CSV ───────────────────────────────────────────────────────────────
# Include empty labeling columns for the HITL step
top100["is_green_human"]      = ""   # human final label (0 / 1)
top100["llm_green_suggested"] = ""   # filled by Part C LLM step
top100["llm_confidence"]      = ""   # low / medium / high
top100["llm_rationale"]       = ""   # 1-3 sentence rationale
top100["human_notes"]         = ""   # optional disagreement notes

export_cols = [
    "doc_id", "text", "p_green", "u",
    "is_green_silver",                # silver label for reference (not used in labeling!)
    "llm_green_suggested", "llm_confidence", "llm_rationale",
    "is_green_human", "human_notes",
]
top100[export_cols].to_csv(OUT_CSV, index=False)
print(f"\nExported {OUT_CSV}  ({len(top100)} rows)")
print(top100[["doc_id", "p_green", "u"]].head(10).to_string(index=False))