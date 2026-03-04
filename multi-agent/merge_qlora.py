"""
merge_qlora.py  –  Merge QLoRA adapter into base model for vLLM serving
Run ONCE from ~/AppliedDL/ before starting the MAS pipeline:
    python multi-agentg/merge_qlora.py
"""

import os
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

BASE_DIR     = "/ceph/home/student.aau.dk/gy53xm/AppliedDL"
BASE_MODEL   = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = os.path.join(BASE_DIR, "qlora-generative-final")
MERGED_PATH  = os.path.join(BASE_DIR, "qlora-merged")

print(f"[INFO] Base dir:    {BASE_DIR}")
print(f"[INFO] Adapter:     {ADAPTER_PATH}")
print(f"[INFO] Output:      {MERGED_PATH}")

# ── 1. Load and merge model ───────────────────────────────────────────────────
print(f"[INFO] Loading base model: {BASE_MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
)

print(f"[INFO] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("[INFO] Merging LoRA weights...")
model = model.merge_and_unload()

print(f"[INFO] Saving merged model to: {MERGED_PATH}")
model.save_pretrained(MERGED_PATH)

# ── 2. Fix tokenizer (avoid TokenizersBackend issue) ─────────────────────────
print("[INFO] Loading and fixing tokenizer...")

# Load from BASE_MODEL directly, not from adapter
# (adapter tokenizer may have TokenizersBackend class set)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=False,    # ← force slow tokenizer, avoids TokenizersBackend
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

tokenizer.save_pretrained(MERGED_PATH)

# ── 3. Patch tokenizer_config.json to remove TokenizersBackend ───────────────
config_path = os.path.join(MERGED_PATH, "tokenizer_config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove or fix the problematic tokenizer_class field
    if config.get("tokenizer_class") == "TokenizersBackend":
        config["tokenizer_class"] = "LlamaTokenizer"
        print("[INFO] Patched tokenizer_class: TokenizersBackend → LlamaTokenizer")

    # Remove tokenizer_file if it points to fast tokenizer
    if "tokenizer_file" in config:
        del config["tokenizer_file"]
        print("[INFO] Removed tokenizer_file field")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

print(f"[INFO] Done! Merged model saved to {MERGED_PATH}")
print(f"       Serve with: vllm --model {MERGED_PATH} --served-model-name qlora-mistral-judge --port 8001")