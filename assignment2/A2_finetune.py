#!/usr/bin/env python
# coding: utf-8

# # Part D: Fine-Tune PatentSBERTa on train_silver + gold_100
#
# Steps:
#   1. Load patents_50k_green.parquet + hitl_green_100.csv
#   2. Build is_green_gold: gold overrides silver for the 100 HITL rows
#   3. Fine-tune PatentSBERTa (sentence-transformers) for binary classification
#      using a SetFit-style or plain SentenceTransformer + classification head approach
#   4. Evaluate on eval_silver  → report Precision / Recall / F1
#   5. Evaluate on gold_100     → report Precision / Recall / F1
#   6. Compare against Part A baseline

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report
import joblib

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME       = "AI-Growth-Lab/PatentSBERTa"
PARQUET_PATH     = "patents_50k_green.parquet"
GOLD_CSV         = "hitl_green_100.csv"
FINETUNED_PATH   = "patentsbert_finetuned"
MAX_SEQ_LENGTH   = 256
EPOCHS           = 1
LR               = 2e-5
BATCH_SIZE       = 16
WARMUP_STEPS     = 50
SEED             = 42

torch.manual_seed(SEED)

# ── 1. Load data ───────────────────────────────────────────────────────────────
df     = pd.read_parquet(PARQUET_PATH)
gold   = pd.read_csv(GOLD_CSV, dtype=str)

# Keep only rows where human has labelled
gold = gold[gold["is_green_human"].notna() & (gold["is_green_human"] != "")].copy()
gold["is_green_human"] = gold["is_green_human"].astype(int)
print(f"Gold-labelled rows available: {len(gold)}")
print(f"  green=1: {(gold['is_green_human']==1).sum()}  green=0: {(gold['is_green_human']==0).sum()}")

# ── 2. Build is_green_gold ─────────────────────────────────────────────────────
# Start with silver label everywhere
df["is_green_gold"] = df["is_green_silver"].astype(int)

# Override with human label for HITL rows (match on text since doc_id is external)
gold_text_to_label = dict(zip(gold["text"].astype(str), gold["is_green_human"]))
mask = df["text"].astype(str).isin(gold_text_to_label)
df.loc[mask, "is_green_gold"] = df.loc[mask, "text"].astype(str).map(gold_text_to_label)
print(f"\nRows updated with gold label: {mask.sum()}")

# ── 3. Build training set: train_silver + gold_100 ─────────────────────────────
train_silver = df[df["split"] == "train_silver"].copy()
eval_silver  = df[df["split"] == "eval_silver"].copy()

# gold_100 as a standalone eval set (rows that were HITL-labelled)
gold_100 = df[mask].copy()
gold_100["label"] = gold_100["is_green_gold"]

# Combined training data
train_combined = pd.concat([train_silver, gold_100], ignore_index=True)
train_combined = train_combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"\nCombined training set: {len(train_combined)} rows")

# ── 4. Fine-tune PatentSBERTa ─────────────────────────────────────────────────
# Strategy: fine-tune the encoder with a contrastive / CosineSimilarity loss,
# then refit the logistic head on the new embeddings.
#
# For binary classification with sentence-transformers we use:
#   - paired examples: (anchor, positive) with label=1 for same class
#                      (anchor, negative) with label=0 for different class
#   - OnlineContrastiveLoss
#
# This is lightweight (1 epoch, ~10k pairs) and works well for domain adaptation.

model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = MAX_SEQ_LENGTH

# Build training pairs for OnlineContrastiveLoss
# Each InputExample: texts=[sent_a, sent_b], label=float (1=similar, 0=dissimilar)
def make_pairs(df_train: pd.DataFrame, n_pairs: int = 20_000, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    pos_idx = df_train.index[df_train["is_green_gold"] == 1].tolist()
    neg_idx = df_train.index[df_train["is_green_gold"] == 0].tolist()

    examples = []
    half = n_pairs // 2

    # positive pairs (same class)
    idxs = rng.choice(pos_idx, size=(half, 2), replace=True)
    for a, b in idxs:
        examples.append(InputExample(
            texts=[str(df_train.at[a, "text"]), str(df_train.at[b, "text"])],
            label=1.0
        ))

    # negative pairs (different class)
    a_idxs = rng.choice(pos_idx, size=half, replace=True)
    b_idxs = rng.choice(neg_idx, size=half, replace=True)
    for a, b in zip(a_idxs, b_idxs):
        examples.append(InputExample(
            texts=[str(df_train.at[a, "text"]), str(df_train.at[b, "text"])],
            label=0.0
        ))

    rng.shuffle(examples)
    return examples

print("\nBuilding training pairs...")
train_examples = make_pairs(train_combined, n_pairs=20_000, seed=SEED)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.OnlineContrastiveLoss(model)

print(f"Fine-tuning {MODEL_NAME} for {EPOCHS} epoch(s)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params={"lr": LR},
    show_progress_bar=True,
    output_path=FINETUNED_PATH,
)
print(f"Fine-tuned model saved to {FINETUNED_PATH}/")

# ── 5. Re-encode and refit classifier head ─────────────────────────────────────
ft_model = SentenceTransformer(FINETUNED_PATH)
ft_model.max_seq_length = MAX_SEQ_LENGTH

def encode(texts, model, batch_size=64):
    return model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

print("\nEncoding train_combined with fine-tuned model...")
X_train_ft = encode(train_combined["text"].astype(str).tolist(), ft_model)
y_train_ft = train_combined["is_green_gold"].astype(int).to_numpy()

print("Encoding eval_silver...")
X_eval_ft  = encode(eval_silver["text"].astype(str).tolist(), ft_model)
y_eval     = eval_silver["is_green_gold"].astype(int).to_numpy()

print("Encoding gold_100...")
X_gold_ft  = encode(gold_100["text"].astype(str).tolist(), ft_model)
y_gold     = gold_100["is_green_gold"].astype(int).to_numpy()

# Fit logistic head (same as baseline for fair comparison)
clf_ft = LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced")
clf_ft.fit(X_train_ft, y_train_ft)
joblib.dump(clf_ft, "logistic_regression_finetuned.joblib")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(X, y_true, label: str, clf):
    y_pred = clf.predict(X)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"\n{'─'*50}")
    print(f"{label}")
    print(f"{'─'*50}")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["not_green", "green"], zero_division=0))

print("\n" + "="*60)
print("EVALUATION RESULTS (Fine-tuned PatentSBERTa)")
print("="*60)
evaluate(X_eval_ft, y_eval, "eval_silver (10k rows)", clf_ft)
evaluate(X_gold_ft, y_gold, "gold_100 (HITL labelled)", clf_ft)

# ── Baseline comparison (re-use saved embeddings if available) ─────────────────
print("\n" + "="*60)
print("BASELINE COMPARISON (frozen PatentSBERTa, from Part A)")
print("="*60)
try:
    clf_base   = joblib.load("logistic_regression_classifier.joblib")
    X_eval_b   = np.load("X_eval_emb.npy")
    y_eval_b   = np.load("y_eval.npy")
    evaluate(X_eval_b, y_eval_b, "eval_silver (baseline)", clf_base)
except FileNotFoundError as e:
    print(f"Baseline artefacts not found ({e}). Run Part A first.")

print("\nDone. Fine-tuned model saved at:", FINETUNED_PATH)#!/usr/bin/env python
# coding: utf-8

# # Part D: Fine-Tune PatentSBERTa on train_silver + gold_100
#
# Steps:
#   1. Load patents_50k_green.parquet + hitl_green_100.csv
#   2. Build is_green_gold: gold overrides silver for the 100 HITL rows
#   3. Fine-tune PatentSBERTa (sentence-transformers) for binary classification
#      using a SetFit-style or plain SentenceTransformer + classification head approach
#   4. Evaluate on eval_silver  → report Precision / Recall / F1
#   5. Evaluate on gold_100     → report Precision / Recall / F1
#   6. Compare against Part A baseline

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report
import joblib

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME       = "AI-Growth-Lab/PatentSBERTa"
PARQUET_PATH     = "patents_50k_green.parquet"
GOLD_CSV         = "hitl_green_100.csv"
FINETUNED_PATH   = "patentsbert_finetuned"
MAX_SEQ_LENGTH   = 256
EPOCHS           = 1
LR               = 2e-5
BATCH_SIZE       = 16
WARMUP_STEPS     = 50
SEED             = 42

torch.manual_seed(SEED)

# ── 1. Load data ───────────────────────────────────────────────────────────────
df     = pd.read_parquet(PARQUET_PATH)
gold   = pd.read_csv(GOLD_CSV, dtype=str)

# Keep only rows where human has labelled
gold = gold[gold["is_green_human"].notna() & (gold["is_green_human"] != "")].copy()
gold["is_green_human"] = gold["is_green_human"].astype(int)
print(f"Gold-labelled rows available: {len(gold)}")
print(f"  green=1: {(gold['is_green_human']==1).sum()}  green=0: {(gold['is_green_human']==0).sum()}")

# ── 2. Build is_green_gold ─────────────────────────────────────────────────────
# Start with silver label everywhere
df["is_green_gold"] = df["is_green_silver"].astype(int)

# Override with human label for HITL rows (match on text since doc_id is external)
gold_text_to_label = dict(zip(gold["text"].astype(str), gold["is_green_human"]))
mask = df["text"].astype(str).isin(gold_text_to_label)
df.loc[mask, "is_green_gold"] = df.loc[mask, "text"].astype(str).map(gold_text_to_label)
print(f"\nRows updated with gold label: {mask.sum()}")

# ── 3. Build training set: train_silver + gold_100 ─────────────────────────────
train_silver = df[df["split"] == "train_silver"].copy()
eval_silver  = df[df["split"] == "eval_silver"].copy()

# gold_100 as a standalone eval set (rows that were HITL-labelled)
gold_100 = df[mask].copy()
gold_100["label"] = gold_100["is_green_gold"]

# Combined training data
train_combined = pd.concat([train_silver, gold_100], ignore_index=True)
train_combined = train_combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"\nCombined training set: {len(train_combined)} rows")

# ── 4. Fine-tune PatentSBERTa ─────────────────────────────────────────────────
# Strategy: fine-tune the encoder with a contrastive / CosineSimilarity loss,
# then refit the logistic head on the new embeddings.
#
# For binary classification with sentence-transformers we use:
#   - paired examples: (anchor, positive) with label=1 for same class
#                      (anchor, negative) with label=0 for different class
#   - OnlineContrastiveLoss
#
# This is lightweight (1 epoch, ~10k pairs) and works well for domain adaptation.

model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = MAX_SEQ_LENGTH

# Build training pairs for OnlineContrastiveLoss
# Each InputExample: texts=[sent_a, sent_b], label=float (1=similar, 0=dissimilar)
def make_pairs(df_train: pd.DataFrame, n_pairs: int = 20_000, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    pos_idx = df_train.index[df_train["is_green_gold"] == 1].tolist()
    neg_idx = df_train.index[df_train["is_green_gold"] == 0].tolist()

    examples = []
    half = n_pairs // 2

    # positive pairs (same class)
    idxs = rng.choice(pos_idx, size=(half, 2), replace=True)
    for a, b in idxs:
        examples.append(InputExample(
            texts=[str(df_train.at[a, "text"]), str(df_train.at[b, "text"])],
            label=1.0
        ))

    # negative pairs (different class)
    a_idxs = rng.choice(pos_idx, size=half, replace=True)
    b_idxs = rng.choice(neg_idx, size=half, replace=True)
    for a, b in zip(a_idxs, b_idxs):
        examples.append(InputExample(
            texts=[str(df_train.at[a, "text"]), str(df_train.at[b, "text"])],
            label=0.0
        ))

    rng.shuffle(examples)
    return examples

print("\nBuilding training pairs...")
train_examples = make_pairs(train_combined, n_pairs=20_000, seed=SEED)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.OnlineContrastiveLoss(model)

print(f"Fine-tuning {MODEL_NAME} for {EPOCHS} epoch(s)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params={"lr": LR},
    show_progress_bar=True,
    output_path=FINETUNED_PATH,
)
print(f"Fine-tuned model saved to {FINETUNED_PATH}/")

# ── 5. Re-encode and refit classifier head ─────────────────────────────────────
ft_model = SentenceTransformer(FINETUNED_PATH)
ft_model.max_seq_length = MAX_SEQ_LENGTH

def encode(texts, model, batch_size=64):
    return model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

print("\nEncoding train_combined with fine-tuned model...")
X_train_ft = encode(train_combined["text"].astype(str).tolist(), ft_model)
y_train_ft = train_combined["is_green_gold"].astype(int).to_numpy()

print("Encoding eval_silver...")
X_eval_ft  = encode(eval_silver["text"].astype(str).tolist(), ft_model)
y_eval     = eval_silver["is_green_gold"].astype(int).to_numpy()

print("Encoding gold_100...")
X_gold_ft  = encode(gold_100["text"].astype(str).tolist(), ft_model)
y_gold     = gold_100["is_green_gold"].astype(int).to_numpy()

# Fit logistic head (same as baseline for fair comparison)
clf_ft = LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced")
clf_ft.fit(X_train_ft, y_train_ft)
joblib.dump(clf_ft, "logistic_regression_finetuned.joblib")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(X, y_true, label: str, clf):
    y_pred = clf.predict(X)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"\n{'─'*50}")
    print(f"{label}")
    print(f"{'─'*50}")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["not_green", "green"], zero_division=0))

print("\n" + "="*60)
print("EVALUATION RESULTS (Fine-tuned PatentSBERTa)")
print("="*60)
evaluate(X_eval_ft, y_eval, "eval_silver (10k rows)", clf_ft)
evaluate(X_gold_ft, y_gold, "gold_100 (HITL labelled)", clf_ft)

# ── Baseline comparison (re-use saved embeddings if available) ─────────────────
print("\n" + "="*60)
print("BASELINE COMPARISON (frozen PatentSBERTa, from Part A)")
print("="*60)
try:
    clf_base   = joblib.load("logistic_regression_classifier.joblib")
    X_eval_b   = np.load("X_eval_emb.npy")
    y_eval_b   = np.load("y_eval.npy")
    evaluate(X_eval_b, y_eval_b, "eval_silver (baseline)", clf_base)
except FileNotFoundError as e:
    print(f"Baseline artefacts not found ({e}). Run Part A first.")

print("\nDone. Fine-tuned model saved at:", FINETUNED_PATH)