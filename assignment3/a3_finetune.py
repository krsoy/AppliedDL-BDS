"""
a3_finetune.py  –  Assignment 3: Fine-tune PatentSBERTa on Silver + Gold
─────────────────────────────────────────────────────────────────────────
Training set:  train_silver (10K) + gold_100 (high-risk HITL-verified, reliable only)
Eval set:      eval_silver (independent held-out)
MAS comparison: loads mas_a3_summary.csv, compares MAS vs LR vs Fine-tuned F1

Run AFTER:
  1. hitl_review.py   → gold_labels_human.json
  2. main.py (A3)     → results_a3/mas_a3_summary.csv
"""

import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    f1_score,
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("=== A3: Fine-tune PatentSBERTa on Silver + Gold Dataset ===\n")

full_df     = pd.read_parquet("../archieved/patents_50k_green.parquet")
pool_df     = pd.read_parquet("../archieved/pool_with_pseudo_labels.parquet").reset_index(drop=True)
pool_sorted = pool_df.sort_values('uncertainty_lr', ascending=False).head(100).reset_index(drop=True)

with open("../archieved/gold_labels_human.json") as f:
    gold_labels = json.load(f)

# ── Build gold dataset ────────────────────────────────────────────────────────
pool_sorted['is_green_gold'] = pool_sorted.index.map(
    lambda i: gold_labels.get(f"claim_{i}", {}).get(
        'is_green_gold', int(pool_sorted.loc[i, 'pseudo_label_lr']))
)
pool_sorted['gold_source'] = pool_sorted.index.map(
    lambda i: gold_labels.get(f"claim_{i}", {}).get('source', 'lr_fallback')
)

# Keep only reliable gold labels (exclude lr_fallback_overflow and lr_fallback)
reliable_sources = {'judge_auto', 'human', 'lr_fallback_skipped'}
gold_reliable = pool_sorted[
    pool_sorted['gold_source'].isin(reliable_sources)
].reset_index(drop=True)

print(f"Gold dataset (reliable only): {len(gold_reliable)} samples")
print(f"Source breakdown:\n{gold_reliable['gold_source'].value_counts().to_string()}")

human_rows = gold_reliable[gold_reliable['gold_source'] == 'human']
if len(human_rows) > 0:
    disagree_lr = (human_rows['is_green_gold'] != human_rows['pseudo_label_lr']).sum()
    print(f"Human vs LR disagreements: {disagree_lr} / {len(human_rows)}")

# ── Define splits ─────────────────────────────────────────────────────────────
silver_df        = full_df[full_df['split'] == 'train_silver']
silver_texts     = silver_df['text'].astype(str).tolist()
silver_labels    = silver_df['is_green_silver'].tolist()

gold_texts       = gold_reliable['text'].astype(str).tolist()
gold_labels_list = gold_reliable['is_green_gold'].tolist()

eval_silver      = full_df[full_df['split'] == 'eval_silver']
eval_texts       = eval_silver['text'].astype(str).tolist()
eval_labels      = eval_silver['is_green_silver'].tolist()

# ── Combined training set: silver + gold ─────────────────────────────────────
combined_texts  = silver_texts  + gold_texts
combined_labels = silver_labels + gold_labels_list

print(f"\ntrain_silver:    {len(silver_texts):>6} samples | green ratio: {sum(silver_labels)/len(silver_labels):.3f}")
print(f"gold (reliable): {len(gold_texts):>6} samples | green ratio: {sum(gold_labels_list)/len(gold_labels_list):.3f}")
print(f"combined:        {len(combined_texts):>6} samples | green ratio: {sum(combined_labels)/len(combined_labels):.3f}")
print(f"eval_silver:     {len(eval_texts):>6} samples | green ratio: {sum(eval_labels)/len(eval_labels):.3f}")

# ── Load original PatentSBERTa ────────────────────────────────────────────────
print("\nLoading PatentSBERTa...")
sbert = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device="cuda")

# ── BEFORE: original PatentSBERTa + train_silver → eval_silver ───────────────
print("\n=== Baseline: Original PatentSBERTa (before fine-tuning) ===")

silver_emb_before = sbert.encode(silver_texts, batch_size=32,
                                  convert_to_numpy=True, show_progress_bar=True)
eval_emb_before   = sbert.encode(eval_texts,   batch_size=32,
                                  convert_to_numpy=True, show_progress_bar=True)

clf_before = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_before.fit(silver_emb_before, silver_labels)
y_pred_before = clf_before.predict(eval_emb_before)

p0, r0, f10, _ = precision_recall_fscore_support(
    eval_labels, y_pred_before, average='binary', zero_division=0)
print(f"[Before] P={p0:.4f}  R={r0:.4f}  F1={f10:.4f}")
print(classification_report(eval_labels, y_pred_before,
                             target_names=['not green', 'green'], zero_division=0))

# ── Build contrastive pairs from COMBINED dataset ─────────────────────────────
print("=== Fine-tuning PatentSBERTa on Silver + Gold ===")

green_texts    = [t for t, l in zip(combined_texts, combined_labels) if l == 1]
nongreen_texts = [t for t, l in zip(combined_texts, combined_labels) if l == 0]
min_len        = min(len(green_texts), len(nongreen_texts))

pair_examples = []
for i in range(min_len):
    pair_examples.append(InputExample(
        texts=[green_texts[i], green_texts[(i+1) % len(green_texts)]],
        label=1.0
    ))
    pair_examples.append(InputExample(
        texts=[green_texts[i % len(green_texts)], nongreen_texts[i]],
        label=0.0
    ))

print(f"Training pairs: {len(pair_examples)} ({min_len} pos + {min_len} neg)")

train_dataloader = DataLoader(pair_examples, shuffle=True, batch_size=16)
train_loss_fn    = losses.CosineSimilarityLoss(sbert)

sbert.fit(
    train_objectives=[(train_dataloader, train_loss_fn)],
    epochs=5,
    warmup_steps=max(1, int(len(pair_examples) * 0.1)),
    output_path="./patentsbert-a3-silver-gold",
    show_progress_bar=True,
)
print("Fine-tuning complete → ./patentsbert-a3-silver-gold")

# ── AFTER: fine-tuned PatentSBERTa + train_silver → eval_silver ──────────────
print("\n=== After Fine-tuning: PatentSBERTa (Silver + Gold) ===")

silver_emb_after = sbert.encode(silver_texts, batch_size=32,
                                 convert_to_numpy=True, show_progress_bar=True)
eval_emb_after   = sbert.encode(eval_texts,   batch_size=32,
                                 convert_to_numpy=True, show_progress_bar=True)

clf_after = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_after.fit(silver_emb_after, silver_labels)
y_pred_after = clf_after.predict(eval_emb_after)

p2, r2, f12, _ = precision_recall_fscore_support(
    eval_labels, y_pred_after, average='binary', zero_division=0)
print(f"[After]  P={p2:.4f}  R={r2:.4f}  F1={f12:.4f}")
print(classification_report(eval_labels, y_pred_after,
                             target_names=['not green', 'green'], zero_division=0))

# ── MAS vs LR vs Fine-tuned comparison ───────────────────────────────────────
print("\n=== MAS vs LR vs Fine-tuned F1 Comparison ===")

mas_path = "../results_a3/mas_a3_summary.csv"
try:
    mas_df = pd.read_csv(mas_path)

    # Align MAS results with eval_silver by text
    # MAS ran on pool_unlabeled (high-risk 100), not eval_silver
    # So we compare MAS final_label vs true_label_lr on the 100 high-risk claims
    valid_mas = mas_df[mas_df['final_label'].isin([0, 1]) &
                       mas_df['true_label_lr'].isin([0, 1])].copy()

    if len(valid_mas) > 0:
        f1_mas = f1_score(valid_mas['true_label_lr'], valid_mas['final_label'],
                          zero_division=0)
        p_mas, r_mas, _, _ = precision_recall_fscore_support(
            valid_mas['true_label_lr'], valid_mas['final_label'],
            average='binary', zero_division=0)

        # LR on the same 100 high-risk claims (true_label_lr IS the LR label, so agreement=1.0)
        # Instead compare LR pseudo labels vs MAS on those 100 claims
        print(f"\nOn {len(valid_mas)} high-risk claims (MAS pool):")
        print(f"  MAS F1 vs LR pseudo labels:  {f1_mas:.4f}  "
              f"(P={p_mas:.4f}, R={r_mas:.4f})")
        agree = (valid_mas['final_label'] == valid_mas['true_label_lr']).mean()
        print(f"  MAS vs LR agreement rate:    {agree:.3f}")

    print(f"\nOn eval_silver ({len(eval_texts)} samples):")
    print(f"  {'Model':<35} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'─'*58}")
    print(f"  {'Original PatentSBERTa + LR':<35} {p0:>7.4f} {r0:>7.4f} {f10:>7.4f}")
    print(f"  {'PatentSBERTa (Silver+Gold) + LR':<35} {p2:>7.4f} {r2:>7.4f} {f12:>7.4f}")
    print(f"  {'Delta':<35} {p2-p0:>+7.4f} {r2-r0:>+7.4f} {f12-f10:>+7.4f}")

except FileNotFoundError:
    print(f"[WARN] {mas_path} not found — run main.py (A3) first.")
    print(f"\nOn eval_silver ({len(eval_texts)} samples):")
    print(f"  {'Model':<35} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'─'*58}")
    print(f"  {'Original PatentSBERTa + LR':<35} {p0:>7.4f} {r0:>7.4f} {f10:>7.4f}")
    print(f"  {'PatentSBERTa (Silver+Gold) + LR':<35} {p2:>7.4f} {r2:>7.4f} {f12:>7.4f}")
    print(f"  {'Delta':<35} {p2-p0:>+7.4f} {r2-r0:>+7.4f} {f12-f10:>+7.4f}")

# ── Save embeddings ───────────────────────────────────────────────────────────
np.save("patentsbert_a3_train_emb.npy", silver_emb_after.astype(np.float32))
np.save("patentsbert_a3_eval_emb.npy",  eval_emb_after.astype(np.float32))
print("\nSaved embeddings → patentsbert_a3_train_emb.npy / patentsbert_a3_eval_emb.npy")
print("\n=== A3 Fine-tuning Complete ===")