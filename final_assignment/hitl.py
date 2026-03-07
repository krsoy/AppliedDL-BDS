"""
hitl_review.py  –  Interactive Human-in-the-Loop Review
────────────────────────────────────────────────────────
Run this script locally (not as sbatch) to manually review
deadlock claims and save your gold labels.

Usage:
    python hitl_review.py
"""

import pandas as pd
import json
import os

MAS_CSV    = "../mas_summary.csv"
POOL_PATH  = "../archieved/pool_with_pseudo_labels.parquet"
GOLD_OUT   = "../gold_labels_human.json"

# ── Load data ─────────────────────────────────────────────────────────────────
mas_df  = pd.read_csv(MAS_CSV)
pool_df = pd.read_parquet(POOL_PATH).reset_index(drop=True)
pool_sorted = pool_df.sort_values('uncertainty_lr', ascending=False).head(100).reset_index(drop=True)

# ── Filter valid MAS results ──────────────────────────────────────────────────
valid_mas = mas_df[~mas_df['rationale'].str.contains('Pipeline error', na=False)].copy()
valid_mas['score_diff'] = abs(valid_mas['advocate_score'] - valid_mas['skeptic_score'])

def categorize(row):
    if row['confidence'] < 0.5 or row['score_diff'] < 0.15:
        return 'deadlock'
    return 'clear'

valid_mas['category'] = valid_mas.apply(categorize, axis=1)

clear_df    = valid_mas[valid_mas['category'] == 'clear']
deadlock_df = valid_mas[valid_mas['category'] == 'deadlock']

print(f"=== HITL Review Session ===")
print(f"Clear claims (auto-accepted): {len(clear_df)}")
print(f"Deadlock claims (need review): {len(deadlock_df)}")
print(f"Token overflow (claims 45-99): 55 → auto LR label")
print(f"\nYou will review {len(deadlock_df)} deadlock claims.")
print("For each claim, enter: 1 (green) or 0 (not green) or s (skip)\n")
input("Press Enter to start review...")

# ── Load previous progress if exists ─────────────────────────────────────────
if os.path.exists(GOLD_OUT):
    with open(GOLD_OUT) as f:
        gold_labels = json.load(f)
    print(f"[INFO] Resuming from previous session ({len(gold_labels)} already reviewed)")
else:
    gold_labels = {}

# ── Auto-accept clear claims ──────────────────────────────────────────────────
for _, row in clear_df.iterrows():
    idx = row['patent_id']
    gold_labels[idx] = {
        'is_green_gold': int(row['final_label']),
        'source': 'judge_auto',
        'confidence': float(row['confidence']),
    }

# ── Interactive review for deadlock claims ────────────────────────────────────
deadlock_list = deadlock_df.to_dict('records')
total = len(deadlock_list)

for i, row in enumerate(deadlock_list):
    idx = row['patent_id']

    # Skip if already reviewed
    if idx in gold_labels and gold_labels[idx]['source'] == 'human':
        continue

    claim_num = int(idx.replace('claim_', ''))
    full_text = pool_sorted.loc[claim_num, 'text'] if claim_num < len(pool_sorted) else row['claim_text']
    lr_label  = int(row['true_label_lr'])

    print(f"\n{'='*70}")
    print(f"[{i+1}/{total}] {idx}  |  LR label: {lr_label}  |  Judge: {row['final_label']}")
    print(f"{'='*70}")
    print(f"\nPATENT CLAIM:\n{full_text[:800]}")
    print(f"\nADVOCATE SCORE: {row['advocate_score']:.2f}  |  SKEPTIC SCORE: {row['skeptic_score']:.2f}")
    print(f"RATIONALE: {row['rationale'][:200]}")
    print(f"\nY02 Categories: E=renewable energy, T=transport, B=buildings,")
    print(f"                A=climate adapt, W=waste, P=low-carbon production")

    while True:
        answer = input("\nYour judgment — 1 (green) / 0 (not green) / s (skip): ").strip().lower()
        if answer in ('0', '1', 's'):
            break
        print("Please enter 0, 1, or s")

    if answer == 's':
        # Skip: fall back to LR label
        gold_labels[idx] = {
            'is_green_gold': lr_label,
            'source': 'lr_fallback_skipped',
            'confidence': 0.5,
        }
    else:
        gold_labels[idx] = {
            'is_green_gold': int(answer),
            'source': 'human',
            'confidence': 1.0,
        }
        if int(answer) != lr_label:
            print(f"  ⚠ You disagreed with LR label ({lr_label})")
        if int(answer) != int(row['final_label']):
            print(f"  ⚠ You disagreed with Judge label ({row['final_label']})")

    # Save progress after each review
    with open(GOLD_OUT, 'w') as f:
        json.dump(gold_labels, f, indent=2)

# ── Handle token overflow claims (45-99): auto LR label ──────────────────────
for i in range(45, 100):
    idx = f"claim_{i}"
    if idx not in gold_labels:
        lr_label = int(pool_sorted.loc[i, 'pseudo_label_lr']) if i < len(pool_sorted) else 0
        gold_labels[idx] = {
            'is_green_gold': lr_label,
            'source': 'lr_fallback_overflow',
            'confidence': 0.5,
        }

# ── Save final gold labels ────────────────────────────────────────────────────
with open(GOLD_OUT, 'w') as f:
    json.dump(gold_labels, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
sources = [v['source'] for v in gold_labels.values()]
print(f"\n=== Review Complete ===")
print(f"Total gold labels:     {len(gold_labels)}")
print(f"Judge auto-accepted:   {sources.count('judge_auto')}")
print(f"Human reviewed:        {sources.count('human')}")
print(f"LR fallback (skipped): {sources.count('lr_fallback_skipped')}")
print(f"LR fallback (overflow):{sources.count('lr_fallback_overflow')}")
print(f"\nSaved → {GOLD_OUT}")
print("Now run part_d_finetune.py to fine-tune PatentSBERTa on gold labels.")