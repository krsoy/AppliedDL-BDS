#!/usr/bin/env python
# coding: utf-8

# ============================================================
# QLoRA Domain Adaptation — Mistral-7B on Green Patent Claims
# Key improvement: model learns to generate yes/no + explanation
# Loss is masked on prompt, only computed on "Answer: ..." part
# ============================================================

import numpy as np
import pandas as pd
import joblib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
from transformers import TrainerCallback

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


# ── 4. QLoRA Domain Adaptation ───────────────────────────────
print("\n=== QLoRA Domain Adaptation ===")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import torch.nn as nn

QLORA_MODEL = "mistralai/Mistral-7B-v0.1"
MAX_LENGTH  = 512

Y02_CONTEXT = (
    "Y02 green technology categories:\n"
    "- Y02E: renewable energy (solar, wind, hydro, fuel cells, smart grid)\n"
    "- Y02T: clean transport (electric vehicles, hydrogen, efficient engines)\n"
    "- Y02B: energy efficiency in buildings (insulation, heat pumps, LED)\n"
    "- Y02A: climate change adaptation (flood protection, drought resistance)\n"
    "- Y02W: waste and recycling (circular economy, biofuels from waste)\n"
    "- Y02P: low-carbon production (green chemistry, carbon capture)\n"
)

def build_prompt(text):
    """Prompt part only — no answer. Used for both train and inference."""
    return (
        "You are a patent classifier specializing in Y02 green technology.\n\n"
        f"{Y02_CONTEXT}\n"
        f"Patent claim: {text}\n\n"
        "Is this a green technology patent? Answer with 'yes' or 'no', and explain why.\n"
        "Answer:"
    )

def build_full_sequence(text, label):
    """Full training sequence = prompt + answer trigger + model fills explanation."""
    prompt = build_prompt(text)
    answer_start = " yes —" if label == 1 else " no —"
    # Model generates the explanation after "yes —" or "no —"
    # During CLM training, the model learns to continue after the answer trigger
    return prompt, answer_start


# ── Tokenize with prompt masking ─────────────────────────────
# Loss is only computed on the "Answer: yes/no — ..." part,
# not on the prompt. This forces the model to learn judgment + reasoning,
# not just memorize the prompt text.

tokenizer = AutoTokenizer.from_pretrained(QLORA_MODEL)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_with_mask(examples):
    input_ids_list = []
    labels_list    = []

    for prompt, answer_start in zip(examples["prompt"], examples["answer_start"]):
        # Tokenize prompt and answer_start separately to find boundary
        prompt_ids       = tokenizer.encode(prompt,       add_special_tokens=True)
        answer_start_ids = tokenizer.encode(answer_start, add_special_tokens=False)

        full_ids = prompt_ids + answer_start_ids

        # Truncate to MAX_LENGTH
        if len(full_ids) > MAX_LENGTH:
            full_ids = full_ids[:MAX_LENGTH]

        # Pad to MAX_LENGTH
        pad_len  = MAX_LENGTH - len(full_ids)
        attn     = [1] * len(full_ids) + [0] * pad_len
        full_ids = full_ids + [tokenizer.pad_token_id] * pad_len

        # Labels: mask prompt with -100, only compute loss on answer part
        prompt_len = min(len(prompt_ids), MAX_LENGTH)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        # Also mask padding
        labels = [
            l if attn[i] == 1 else -100
            for i, l in enumerate(labels)
        ]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    return {
        "input_ids":      input_ids_list,
        "attention_mask": [[1 if x != tokenizer.pad_token_id else 0
                            for x in ids] for ids in input_ids_list],
        "labels":         labels_list,
    }


# ── 准备数据集 ───────────────────────────────────────────────
train_silver_df = df[df["split"] == "train_silver"]
eval_silver_df  = df[df["split"] == "eval_silver"]

print(f"Train silver: {len(train_silver_df)} | "
      f"green ratio: {train_silver_df['is_green_silver'].mean():.3f}")
print(f"Eval silver:  {len(eval_silver_df)}  | "
      f"green ratio: {eval_silver_df['is_green_silver'].mean():.3f}")

def prepare_dataset(df):
    rows = []
    for _, row in df.iterrows():
        prompt, answer_start = build_full_sequence(row["text"], row["is_green_silver"])
        rows.append({"prompt": prompt, "answer_start": answer_start})
    return Dataset.from_list(rows)

train_dataset = prepare_dataset(train_silver_df)
eval_dataset  = prepare_dataset(eval_silver_df)

train_dataset = train_dataset.map(tokenize_with_mask, batched=True,
                                   remove_columns=["prompt", "answer_start"])
eval_dataset  = eval_dataset.map(tokenize_with_mask,  batched=True,
                                   remove_columns=["prompt", "answer_start"])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")


# ── 4-bit 量化 ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    QLORA_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.eos_token_id
model = prepare_model_for_kbit_training(model)

# ── LoRA 配置 ────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
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
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    report_to="none"
)

# Use default data collator — labels already set with -100 masking
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[NaNStopCallback()],
)

trainer.train()

model.save_pretrained("./qlora-generative-final")
tokenizer.save_pretrained("./qlora-generative-final")
print("QLoRA 训练完成，已保存到 ./qlora-generative-final")


# ── 5. QLoRA 对 pool_unlabeled 伪标注 ────────────────────────
print("\n=== QLoRA 伪标注 pool_unlabeled ===")

model.eval()
pool_texts = pool_df["text"].astype(str).tolist()
BATCH_SIZE = 8
all_probs  = []

# Get token ids for " yes" and " no"
yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
no_token_id  = tokenizer.encode(" no",  add_special_tokens=False)[0]

print(f"yes token id: {yes_token_id}, no token id: {no_token_id}")

with torch.no_grad():
    for i in range(0, len(pool_texts), BATCH_SIZE):
        batch_texts = pool_texts[i:i+BATCH_SIZE]
        prompts = [build_prompt(t) for t in batch_texts]

        enc_batch = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        enc_batch = {k: v.to(model.device) for k, v in enc_batch.items()}

        outputs = model(**enc_batch)

        # Compare P(yes) vs P(no) at the next-token position
        last_logits   = outputs.logits[:, -1, :]
        yes_no_logits = last_logits[:, [yes_token_id, no_token_id]]
        probs         = torch.softmax(yes_no_logits, dim=-1)[:, 0]  # P(yes)

        all_probs.extend(probs.cpu().float().numpy())

        if i % (BATCH_SIZE * 20) == 0:
            print(f"  推理进度: {i}/{len(pool_texts)}")

prob_pool_qlora        = np.array(all_probs, dtype=np.float32)
pseudo_label_qlora     = (prob_pool_qlora >= 0.5).astype(int)
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