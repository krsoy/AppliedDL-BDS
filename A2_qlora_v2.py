#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 接续已有结果，直接从保存的文件加载
# ============================================================

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
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

X_train_emb       = np.load("X_train_emb.npy")
X_eval_emb        = np.load("X_eval_emb.npy")
y_train           = np.load("y_train.npy")
y_eval            = np.load("y_eval.npy")
prob_train        = np.load("prob_train.npy")
prob_eval         = np.load("prob_eval.npy")
uncertainty_train = np.load("uncertainty_train.npy")
uncertainty_eval  = np.load("uncertainty_eval.npy")
clf               = joblib.load("logistic_regression_classifier.joblib")

print(f"X_train_emb: {X_train_emb.shape}, X_eval_emb: {X_eval_emb.shape}")

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

# 释放显存
del enc
torch.cuda.empty_cache()
print("已释放 SentenceTransformer 显存")


# ── 3. LR 对 pool 伪标注（基线） ─────────────────────────────
print("\n=== LR 伪标注 pool_unlabeled ===")

prob_pool        = clf.predict_proba(X_pool_emb)[:, 1]
pseudo_label_lr  = (prob_pool >= 0.5).astype(int)
uncertainty_pool = 1 - 2 * np.abs(prob_pool - 0.5)

pool_df["pseudo_prob_lr"]  = prob_pool
pool_df["pseudo_label_lr"] = pseudo_label_lr
pool_df["uncertainty_lr"]  = uncertainty_pool

np.save("prob_pool.npy",        prob_pool.astype(np.float32))
np.save("uncertainty_pool.npy", uncertainty_pool.astype(np.float32))

print(f"LR pseudo green=1 比例: {pseudo_label_lr.mean():.3f}")
print(f"高置信度样本(>0.9 or <0.1): {((prob_pool>0.9)|(prob_pool<0.1)).sum()}")


# ── 4. QLoRA 生成式微调 ───────────────────────────────────────
print("\n=== QLoRA 微调 (Generative) ===")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,          # ← 改为生成式
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

QLORA_MODEL = "mistralai/Mistral-7B-v0.1"
MAX_LENGTH  = 512

# ── Prompt 格式 ──────────────────────────────────────────────
def format_prompt(text, label=None):
    prompt = (
        "You are a patent classifier. "
        "Determine if the following patent claim relates to green technology (Y02 classification).\n\n"
        f"Patent claim: {text}\n\n"
        "Is this a green technology patent? Answer with 'yes' or 'no'.\n"
        "Answer:"
    )
    if label is not None:
        answer = " yes" if label == 1 else " no"
        return prompt + answer
    return prompt

# ── 准备数据集 ───────────────────────────────────────────────
train_silver_df = df[df["split"] == "train_silver"]
eval_silver_df  = df[df["split"] == "eval_silver"]

def prepare_dataset(df):
    return Dataset.from_list([
        {"text": format_prompt(row["text"], row["is_green_silver"])}
        for _, row in df.iterrows()
    ])

train_dataset = prepare_dataset(train_silver_df)
eval_dataset  = prepare_dataset(eval_silver_df)

# ── Tokenizer ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(QLORA_MODEL)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(batch):
    result = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    result["labels"] = result["input_ids"].copy()  # CLM: 预测下一个token
    return result

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
eval_dataset  = eval_dataset.map(tokenize,  batched=True, remove_columns=["text"])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# ── 4-bit 量化 ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(   # ← 生成式模型
    QLORA_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.eos_token_id
model = prepare_model_for_kbit_training(model)

# ── LoRA 配置 ────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,              # ← 改为 CAUSAL_LM
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── NaN 早停 callback ────────────────────────────────────────
class NaNStopCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss", None)
        if loss is not None and (str(loss) == "nan" or float(loss) != float(loss)):
            print("❌ loss=nan，立即停止")
            control.should_training_stop = True

# ── 训练参数 ─────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./qlora-generative",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # 等效 batch=16
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,           # loss 越小越好
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[NaNStopCallback()],
)

trainer.train()

# 保存模型
model.save_pretrained("./qlora-generative-final")
tokenizer.save_pretrained("./qlora-generative-final")
print("QLoRA 训练完成，已保存到 ./qlora-generative-final")


# ── 5. QLoRA 对 pool_unlabeled 伪标注 ────────────────────────
print("\n=== QLoRA 伪标注 pool_unlabeled ===")

model.eval()
pool_texts = pool_df["text"].astype(str).tolist()
BATCH_SIZE = 16   # 生成模式显存占用更大，batch 要小
all_probs  = []

# 获取 yes/no 对应的 token id
yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
no_token_id  = tokenizer.encode(" no",  add_special_tokens=False)[0]

with torch.no_grad():
    for i in range(0, len(pool_texts), BATCH_SIZE):
        batch_texts = pool_texts[i:i+BATCH_SIZE]
        prompts = [format_prompt(t) for t in batch_texts]  # 不带答案

        enc_batch = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        enc_batch = {k: v.to(model.device) for k, v in enc_batch.items()}

        outputs = model(**enc_batch)

        # 取最后一个位置的 logits，比较 yes vs no 的概率
        last_logits = outputs.logits[:, -1, :]  # (batch, vocab)
        yes_no_logits = last_logits[:, [yes_token_id, no_token_id]]
        probs = torch.softmax(yes_no_logits, dim=-1)[:, 0]  # P(yes)

        all_probs.extend(probs.cpu().float().numpy())

        if i % (BATCH_SIZE * 20) == 0:
            print(f"  推理进度: {i}/{len(pool_texts)}")

prob_pool_qlora    = np.array(all_probs, dtype=np.float32)
pseudo_label_qlora = (prob_pool_qlora >= 0.5).astype(int)
uncertainty_pool_qlora = 1 - 2 * np.abs(prob_pool_qlora - 0.5)

pool_df["pseudo_prob_qlora"]  = prob_pool_qlora
pool_df["pseudo_label_qlora"] = pseudo_label_qlora
pool_df["uncertainty_qlora"]  = uncertainty_pool_qlora

np.save("prob_pool_qlora.npy",        prob_pool_qlora)
np.save("uncertainty_pool_qlora.npy", uncertainty_pool_qlora)

print(f"QLoRA pseudo green=1 比例: {pseudo_label_qlora.mean():.3f}")
print(f"高置信度样本(>0.9 or <0.1): {((prob_pool_qlora>0.9)|(prob_pool_qlora<0.1)).sum()}")


# ── 6. LR vs QLoRA 伪标注对比 ────────────────────────────────
print("\n=== LR vs QLoRA 伪标注一致性 ===")

agree = (pseudo_label_lr == pseudo_label_qlora).mean()
print(f"两模型标注一致率: {agree:.3f}")

HIGH_CONF = 0.85
mask_conf = (
    ((prob_pool >= HIGH_CONF) | (prob_pool <= 1 - HIGH_CONF)) &
    ((prob_pool_qlora >= HIGH_CONF) | (prob_pool_qlora <= 1 - HIGH_CONF))
)
pool_df["high_conf_both"] = mask_conf
print(f"双模型高置信度样本: {mask_conf.sum()} / {len(pool_df)}")


# ── 7. 保存最终结果 ───────────────────────────────────────────
pool_df.to_parquet("pool_with_pseudo_labels.parquet", index=False)
print("\n已保存 pool_with_pseudo_labels.parquet")
print(pool_df[["pseudo_label_lr", "pseudo_label_qlora", "high_conf_both"]].describe())