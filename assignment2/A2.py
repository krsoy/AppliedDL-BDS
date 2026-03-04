#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import itertools
import torch
from sentence_transformers import SentenceTransformer

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# # default hyperparameters for the lecture

# In[2]:


NUM_VIRTUAL_TOKENS = 8     # prompt tuning parameters (small)
MAX_LENGTH = 256           # sequence length for training
TRAIN_SAMPLES = 2000       # how many claims to stream in for training
EVAL_SAMPLES = 200         # small held-out set for quick sanity check
MAX_STEPS = 80             # for lecture speed; increase for better results

SEED = 42


# # loading the dataset

# In[23]:


dataset_name = "AI-Growth-Lab/patents_claims_1.5m_traim_test"

# download full data
train_ds = load_dataset(dataset_name, split="train")
test_ds = load_dataset(dataset_name, split="test")

def take_text(stream, n):
    out = []
    for ex in itertools.islice(stream, n):
        # Keep just claim text for language modeling
        out.append({"text": ex["text"]})
    return Dataset.from_list(out)


# train_ds_raw = take_text(train_stream, TRAIN_SAMPLES)
# eval_ds_raw  = take_text(test_stream,  EVAL_SAMPLES)

train_text = train_ds.select_columns(["text"])
test_text  = test_ds.select_columns(["text"])




# In[25]:


train_ds


# In[37]:


train_ds_select = train_ds.with_format("pandas")
# filter columns have 'Y02' in it
for df in train_ds_select.iter(batch_size=50000):
     print(df.filter(like="Y02", axis=1).sum(axis=1).sum())
     # it contains Y02 like column and summary is >=1, so mark as True, else False.
     # keep top 25K True rows and 25K False rows to balance the dataset.
     # storage in patents_50k_green.parquet



# In[38]:


import pandas as pd
TARGET_POS = 25_000
TARGET_NEG = 25_000
BATCH_SIZE = 50_000

# pick the minimal columns to save
# include at least 'text' + label; add ids if you have them
KEEP_COLS = ["text"]  # add e.g. "patent_id", "publication_number" if present

train_ds_select = train_ds.with_format("pandas")

pos_parts = []
neg_parts = []
pos_cnt = 0
neg_cnt = 0

for df in train_ds_select.iter(batch_size=BATCH_SIZE):
    # 1) Extract Y02 columns by column-name match
    y02_cols_df = df.filter(like="Y02", axis=1)

    # If there are no Y02 columns in schema, this approach can't work
    if y02_cols_df.shape[1] == 0:
        raise ValueError("No columns containing 'Y02' found. Are CPC codes stored as values instead of one-hot columns?")

    # 2) Make sure they are numeric (0/1), then sum per row
    y02_block = y02_cols_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    y02_count = y02_block.sum(axis=1)

    # 3) Row label: green if any Y02 column is active
    is_green = (y02_count >= 1)

    # 4) Add silver label column
    df = df.copy()
    df["is_green_silver"] = is_green.astype("int8")

    # 5) Take only what you want to store
    # (keep label + any metadata cols)
    cols_to_store = [c for c in KEEP_COLS if c in df.columns] + ["is_green_silver"]
    df_small = df[cols_to_store]

    # 6) Fill buckets until reaching 25k/25k
    if pos_cnt < TARGET_POS:
        take_pos = df_small[df_small["is_green_silver"] == 1]
        need = TARGET_POS - pos_cnt
        if len(take_pos) > need:
            take_pos = take_pos.iloc[:need]
        if len(take_pos) > 0:
            pos_parts.append(take_pos)
            pos_cnt += len(take_pos)

    if neg_cnt < TARGET_NEG:
        take_neg = df_small[df_small["is_green_silver"] == 0]
        need = TARGET_NEG - neg_cnt
        if len(take_neg) > need:
            take_neg = take_neg.iloc[:need]
        if len(take_neg) > 0:
            neg_parts.append(take_neg)
            neg_cnt += len(take_neg)

    # 7) Stop early when done
    if pos_cnt >= TARGET_POS and neg_cnt >= TARGET_NEG:
        break

# 8) Build final 50k dataframe
pos_df = pd.concat(pos_parts, ignore_index=True) if pos_parts else pd.DataFrame(columns=cols_to_store)
neg_df = pd.concat(neg_parts, ignore_index=True) if neg_parts else pd.DataFrame(columns=cols_to_store)

balanced_df = pd.concat([pos_df, neg_df], ignore_index=True)

# Optional shuffle so it's not all positives then negatives
balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 9) Add split label if you want (here: all "train_silver" + "pool_unlabeled" doesn't apply since both are labeled;
# but you can encode as "train_silver" for positives and "pool_unlabeled" for negatives if you insist)
balanced_df["split"] = balanced_df["is_green_silver"].map({1: "train_silver", 0: "pool_unlabeled"})

# 10) Save
balanced_df.to_parquet("patents_50k_green.parquet", index=False)

print("Saved patents_50k_green.parquet")
print("pos:", (balanced_df["is_green_silver"] == 1).sum(), "neg:", (balanced_df["is_green_silver"] == 0).sum())


# In[39]:


import pandas as pd

SEED = 42

EVAL_PER_CLASS = 5_000
TRAIN_SILVER_PER_CLASS = 5_000
# the remainder goes to pool_unlabeled automatically

df = pd.read_parquet("patents_50k_green.parquet")

# shuffle first for randomness
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

pos = df[df["is_green_silver"] == 1].copy()
neg = df[df["is_green_silver"] == 0].copy()

assert len(pos) >= (EVAL_PER_CLASS + TRAIN_SILVER_PER_CLASS), "Not enough positive samples for requested split sizes."
assert len(neg) >= (EVAL_PER_CLASS + TRAIN_SILVER_PER_CLASS), "Not enough negative samples for requested split sizes."

# assign eval_silver
# first 5000 positives and 5000 negatives go to eval_silver;
pos_eval = pos.iloc[:EVAL_PER_CLASS].copy()
neg_eval = neg.iloc[:EVAL_PER_CLASS].copy()
# the rest will be split into train_silver and pool_unlabeled
pos_rest = pos.iloc[EVAL_PER_CLASS:].copy() # remaining positives after taking eval, this is where we will split train_silver vs pool_unlabeled
neg_rest = neg.iloc[EVAL_PER_CLASS:].copy() # remaining negatives after taking eval

# assign train_silver
pos_train = pos_rest.iloc[:TRAIN_SILVER_PER_CLASS].copy() # next 5000 positives go to train_silver
neg_train = neg_rest.iloc[:TRAIN_SILVER_PER_CLASS].copy() # next 5000 negatives go to train_silver
pos_pool  = pos_rest.iloc[TRAIN_SILVER_PER_CLASS:].copy() # any remaining positives go to pool_unlabeled
neg_pool  = neg_rest.iloc[TRAIN_SILVER_PER_CLASS:].copy() # any remaining negatives go to pool_unlabeled

pos_eval["split"]  = "eval_silver"
neg_eval["split"]  = "eval_silver"
pos_train["split"] = "train_silver"
neg_train["split"] = "train_silver"
pos_pool["split"]  = "pool_unlabeled"
neg_pool["split"]  = "pool_unlabeled"

out = pd.concat([pos_eval, neg_eval, pos_train, neg_train, pos_pool, neg_pool], ignore_index=True)
out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

out.to_parquet("patents_50k_green.parquet", index=False)

print(out["split"].value_counts())
print("eval pos/neg:", out.query("split=='eval_silver'")["is_green_silver"].value_counts().to_dict())
print("train pos/neg:", out.query("split=='train_silver'")["is_green_silver"].value_counts().to_dict())
print("pool pos/neg:", out.query("split=='pool_unlabeled'")["is_green_silver"].value_counts().to_dict())


# # loading the model

# In[40]:


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"

df = pd.read_parquet("patents_50k_green.parquet")

train = df[df["split"] == "train_silver"]
eval_ = df[df["split"] == "eval_silver"]

X_train_text = train["text"].astype(str).tolist()
y_train = train["is_green_silver"].astype(int).to_numpy()

X_eval_text = eval_["text"].astype(str).tolist()
y_eval = eval_["is_green_silver"].astype(int).to_numpy()

# Frozen encoder: we only train a classifier on top of embeddings
enc = SentenceTransformer(MODEL_NAME,device="cuda")

X_train_emb = enc.encode(X_train_text, batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
X_eval_emb  = enc.encode(X_eval_text, batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)

clf = LogisticRegression(max_iter=10,n_jobs=-1, class_weight="balanced")
clf.fit(X_train_emb, y_train)

y_pred = clf.predict(X_eval_emb)

p, r, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average="binary", zero_division=0)
print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")


# 保存
np.save("X_train_emb.npy", X_train_emb.astype(np.float32))
np.save("X_eval_emb.npy",  X_eval_emb.astype(np.float32))

# （可选）也存标签，训练/评估直接复用
np.save("y_train.npy", y_train.astype(np.int64))
np.save("y_eval.npy",  y_eval.astype(np.int64))

# save the trained classifier for later use (e.g. via joblib)
import joblib
joblib.dump(clf, "logistic_regression_classifier.joblib")

#storage the probability scores for later use (e.g. for threshold tuning or analysis)
prob_train = clf.predict_proba(X_train_emb)[:, 1]  # probability of positive class
prob_eval = clf.predict_proba(X_eval_emb)[:, 1]
np.save("prob_train.npy", prob_train.astype(np.float32))
np.save("prob_eval.npy", prob_eval.astype(np.float32))

# calculate u = 1 − 2 · |p − 0.5| uncertainty score based on probability (higher means more uncertain)
uncertainty_eval = 1 - 2 * np.abs(prob_eval - 0.5)
np.save("uncertainty_eval.npy", uncertainty_eval.astype(np.float32))
uncertainty_train = 1 - 2 * np.abs(prob_train - 0.5)
np.save("uncertainty_train.npy", uncertainty_train.astype(np.float32))


