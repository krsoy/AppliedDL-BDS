"""
part_d_finetune.py  –  Fine-tune PatentSBERTa on Gold Dataset
──────────────────────────────────────────────────────────────
Run AFTER hitl_review.py has produced gold_labels_human.json

Baseline (before): original PatentSBERTa + train_silver → eval_silver
After:             gold fine-tuned PatentSBERTa + train_silver → eval_silver
"""

import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report

# ── Load data ─────────────────────────────────────────────────────────────────
print("=== Part D: Fine-tune PatentSBERTa on Gold Dataset ===\n")

full_df     = pd.read_parquet("../patents_50k_green.parquet")
pool_df     = pd.read_parquet("../pool_with_pseudo_labels.parquet").reset_index(drop=True)
pool_sorted = pool_df.sort_values('uncertainty_lr', ascending=False).head(100).reset_index(drop=True)

with open("../gold_labels_human.json") as f:
    gold_labels = json.load(f)

# ── Build gold dataset ────────────────────────────────────────────────────────
pool_sorted['is_green_gold'] = pool_sorted.index.map(
    lambda i: gold_labels.get(f"claim_{i}", {}).get(
        'is_green_gold', int(pool_sorted.loc[i, 'pseudo_label_lr']))
)
pool_sorted['gold_source'] = pool_sorted.index.map(
    lambda i: gold_labels.get(f"claim_{i}", {}).get('source', 'lr_fallback')
)

print(f"Gold dataset (all): {len(pool_sorted)} samples")
print(f"\nSource breakdown (before filter):\n{pool_sorted['gold_source'].value_counts().to_string()}")

# Keep only reliable labels:
#   judge_auto        — Judge gave a clear confident decision
#   human             — you manually reviewed and labeled
#   lr_fallback_skipped — you chose to skip (LR used as proxy)
# Exclude:
#   lr_fallback_overflow  — claims 45-99, token overflow, never processed by MAS
#   lr_fallback           — catch-all fallback, no MAS signal at all
reliable_sources = {'judge_auto', 'human', 'lr_fallback_skipped'}
pool_sorted = pool_sorted[
    pool_sorted['gold_source'].isin(reliable_sources)
].reset_index(drop=True)

print(f"\nGold dataset (after filtering unreliable): {len(pool_sorted)} samples")
print(f"is_green_gold:\n{pool_sorted['is_green_gold'].value_counts().to_string()}")
print(f"\nSource breakdown (after filter):\n{pool_sorted['gold_source'].value_counts().to_string()}")

human_rows = pool_sorted[pool_sorted['gold_source'] == 'human']
if len(human_rows) > 0:
    disagree_lr = (human_rows['is_green_gold'] != human_rows['pseudo_label_lr']).sum()
    print(f"\nHuman vs LR disagreements: {disagree_lr} / {len(human_rows)}")

pool_sorted.to_parquet("../gold_dataset.parquet", index=False)
print("\nSaved ../gold_dataset.parquet")

# ── Define splits ─────────────────────────────────────────────────────────────
silver_df     = full_df[full_df['split'] == 'train_silver']
silver_texts  = silver_df['text'].astype(str).tolist()
silver_labels = silver_df['is_green_silver'].tolist()

eval_silver   = full_df[full_df['split'] == 'eval_silver']
eval_texts    = eval_silver['text'].astype(str).tolist()
eval_labels   = eval_silver['is_green_silver'].tolist()

gold_texts    = pool_sorted['text'].astype(str).tolist()
gold_labels_list = pool_sorted['is_green_gold'].tolist()

print(f"\ntrain_silver: {len(silver_texts)} samples | green ratio: {sum(silver_labels)/len(silver_labels):.3f}")
print(f"gold:         {len(gold_texts)} samples | green ratio: {sum(gold_labels_list)/len(gold_labels_list):.3f}")
print(f"eval_silver:  {len(eval_texts)} samples | green ratio: {sum(eval_labels)/len(eval_labels):.3f}")

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

p, r, f1, _ = precision_recall_fscore_support(
    eval_labels, y_pred_before, average='binary', zero_division=0)
print(f"[Before] P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
print(classification_report(eval_labels, y_pred_before,
                             target_names=['not green', 'green'], zero_division=0))

# ── Fine-tune on gold dataset ─────────────────────────────────────────────────
print("=== Fine-tuning PatentSBERTa on Gold Dataset ===")

green_texts    = [t for t, l in zip(gold_texts, gold_labels_list) if l == 1]
nongreen_texts = [t for t, l in zip(gold_texts, gold_labels_list) if l == 0]
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

train_dataloader = DataLoader(pair_examples, shuffle=True, batch_size=8)
train_loss       = losses.CosineSimilarityLoss(sbert)

sbert.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=max(1, int(len(pair_examples) * 0.1)),
    output_path="./patentsbert-gold-finetuned",
    show_progress_bar=True,
)
print("Fine-tuning complete → ./patentsbert-gold-finetuned")

# ── AFTER: fine-tuned PatentSBERTa + train_silver → eval_silver ──────────────
print("\n=== After Fine-tuning: PatentSBERTa + Gold ===")

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

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'Metric':<12} {'Before':>8} {'After':>8} {'Delta':>8}")
print(f"{'─'*42}")
print(f"{'Precision':<12} {p:>8.4f} {p2:>8.4f} {p2-p:>+8.4f}")
print(f"{'Recall':<12} {r:>8.4f} {r2:>8.4f} {r2-r:>+8.4f}")
print(f"{'F1':<12} {f1:>8.4f} {f12:>8.4f} {f12-f1:>+8.4f}")

np.save("patentsbert_gold_train_emb.npy", silver_emb_after.astype(np.float32))
np.save("patentsbert_gold_eval_emb.npy",  eval_emb_after.astype(np.float32))
print("\nSaved embeddings.")
print("\n=== Part D Complete ===")