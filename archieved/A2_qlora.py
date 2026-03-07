#!/usr/bin/env python
# coding: utf-8
from gc import callbacks

# ============================================================
# 接续已有结果，直接从保存的文件加载
# ============================================================

import numpy as np
import pandas as pd
import joblib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from transformers import TrainerCallback

# 检查 GPU 可用性
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found! This script requires CUDA.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# ── 1. 加载已保存的结果 ──────────────────────────────────────
print("=== 加载已保存文件 ===")

df = pd.read_parquet("patents_50k_green.parquet")
print(df["split"].value_counts())

X_train_emb = np.load("X_train_emb.npy")
X_eval_emb  = np.load("X_eval_emb.npy")
y_train     = np.load("y_train.npy")
y_eval      = np.load("y_eval.npy")
prob_train  = np.load("prob_train.npy")
prob_eval   = np.load("prob_eval.npy")
uncertainty_train = np.load("uncertainty_train.npy")
uncertainty_eval  = np.load("uncertainty_eval.npy")
clf         = joblib.load("logistic_regression_classifier.joblib")

print(f"X_train_emb: {X_train_emb.shape}, X_eval_emb: {X_eval_emb.shape}")

# 验证 eval 性能
y_pred = clf.predict(X_eval_emb)
p, r, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average="binary", zero_division=0)
print(f"[LR Baseline] Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")


# ── 2. 对 pool_unlabeled 编码 ────────────────────────────────
print("\n=== 编码 pool_unlabeled ===")

pool_df = df[df["split"] == "pool_unlabeled"].copy().reset_index(drop=True)
print(f"pool size: {len(pool_df)}")

enc = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device="cuda")

X_pool_emb = enc.encode(
    pool_df["text"].astype(str).tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
np.save("X_pool_emb.npy", X_pool_emb.astype(np.float32))
print(f"X_pool_emb: {X_pool_emb.shape}")


# ── 3. LR 对 pool 伪标注（基线） ─────────────────────────────
print("\n=== LR 伪标注 pool_unlabeled ===")

prob_pool = clf.predict_proba(X_pool_emb)[:, 1]
pseudo_label_lr = (prob_pool >= 0.5).astype(int)
uncertainty_pool = 1 - 2 * np.abs(prob_pool - 0.5)

pool_df["pseudo_prob_lr"]   = prob_pool
pool_df["pseudo_label_lr"]  = pseudo_label_lr
pool_df["uncertainty_lr"]   = uncertainty_pool

np.save("prob_pool.npy", prob_pool.astype(np.float32))
np.save("uncertainty_pool.npy", uncertainty_pool.astype(np.float32))

print(f"LR pseudo green=1 比例: {pseudo_label_lr.mean():.3f}")
print(f"高置信度样本(>0.9 or <0.1): {((prob_pool>0.9)|(prob_pool<0.1)).sum()}")


# ── 4. QLoRA 微调 ────────────────────────────────────────────
print("\n=== QLoRA 微调 ===")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

QLORA_MODEL = "mistralai/Mistral-7B-v0.1"  # 显存充足可换 mistralai/Mistral-7B-v0.1
MAX_LENGTH  = 256

train_silver_df = df[df["split"] == "train_silver"][["text", "is_green_silver"]].rename(
    columns={"text": "claim_text", "is_green_silver": "label"}
)
eval_silver_df = df[df["split"] == "eval_silver"][["text", "is_green_silver"]].rename(
    columns={"text": "claim_text", "is_green_silver": "label"}
)

train_dataset = Dataset.from_pandas(train_silver_df.reset_index(drop=True))
eval_dataset  = Dataset.from_pandas(eval_silver_df.reset_index(drop=True))

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(QLORA_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(batch):
    return tokenizer(
        batch["claim_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["claim_text"])
eval_dataset  = eval_dataset.map(tokenize,  batched=True, remove_columns=["claim_text"])
train_dataset = train_dataset.rename_column("label", "labels")
eval_dataset  = eval_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    QLORA_MODEL,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto"
)
nn.init.normal_(model.score.weight, mean=0.0, std=0.02)


model.config.pad_token_id = tokenizer.eos_token_id  # ← 加这里

model = prepare_model_for_kbit_training(model)

# LoRA 配置（distilbert 的 attention 层名称）
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Mistral 的 attention 层
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}

training_args = TrainingArguments(
    output_dir="./qlora-green-patent-4",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=5e-5,        # ← 关键修复
    max_grad_norm=0.3,         # ← 防梯度爆炸
    warmup_ratio=0.03,         # ← 稳定开始
    bf16=True,
    logging_steps=1,
    report_to="none"
)
class NaNStopCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss", None)
        if loss is not None and (str(loss) == "nan" or float(loss) != float(loss)):
            print("❌ loss=nan，立即停止")
            control.should_training_stop = True
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[NaNStopCallback()],
)

trainer.train()
print("QLoRA 训练完成")


# ── 5. QLoRA 对 pool_unlabeled 伪标注 ───────────────────────
print("\n=== QLoRA 伪标注 pool_unlabeled ===")

model.eval()

pool_texts = pool_df["text"].astype(str).tolist()
BATCH_SIZE = 64
all_probs = []

with torch.no_grad():
    for i in range(0, len(pool_texts), BATCH_SIZE):
        batch_texts = pool_texts[i:i+BATCH_SIZE]
        enc_batch = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        enc_batch = {k: v.to(model.device) for k, v in enc_batch.items()}
        outputs = model(**enc_batch)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().float().numpy())
        if i % (BATCH_SIZE * 20) == 0:
            print(f"  推理进度: {i}/{len(pool_texts)}")

prob_pool_qlora = np.array(all_probs, dtype=np.float32)
pseudo_label_qlora = (prob_pool_qlora >= 0.5).astype(int)
uncertainty_pool_qlora = 1 - 2 * np.abs(prob_pool_qlora - 0.5)

pool_df["pseudo_prob_qlora"]  = prob_pool_qlora
pool_df["pseudo_label_qlora"] = pseudo_label_qlora
pool_df["uncertainty_qlora"]  = uncertainty_pool_qlora

np.save("prob_pool_qlora.npy", prob_pool_qlora)
np.save("uncertainty_pool_qlora.npy", uncertainty_pool_qlora)

print(f"QLoRA pseudo green=1 比例: {pseudo_label_qlora.mean():.3f}")
print(f"高置信度样本(>0.9 or <0.1): {((prob_pool_qlora>0.9)|(prob_pool_qlora<0.1)).sum()}")


# ── 6. LR vs QLoRA 伪标注对比 ───────────────────────────────
print("\n=== LR vs QLoRA 伪标注一致性 ===")

agree = (pseudo_label_lr == pseudo_label_qlora).mean()
print(f"两模型标注一致率: {agree:.3f}")

# 仅保留两模型都高置信度的样本（更可靠的伪标签）
HIGH_CONF = 0.85
mask_conf = (
    ((prob_pool >= HIGH_CONF) | (prob_pool <= 1 - HIGH_CONF)) &
    ((prob_pool_qlora >= HIGH_CONF) | (prob_pool_qlora <= 1 - HIGH_CONF))
)
pool_df["high_conf_both"] = mask_conf
print(f"双模型高置信度样本: {mask_conf.sum()} / {len(pool_df)}")


# ── 7. 保存最终结果 ──────────────────────────────────────────
pool_df.to_parquet("pool_with_pseudo_labels.parquet", index=False)
print("\n已保存 pool_with_pseudo_labels.parquet")
print(pool_df[["pseudo_label_lr", "pseudo_label_qlora", "high_conf_both"]].describe())