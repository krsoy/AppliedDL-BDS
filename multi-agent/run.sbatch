#!/bin/bash
#SBATCH --job-name=green_patent_mas
#SBATCH --output=/ceph/home/student.aau.dk/gy53xm/AppliedDL/logs/mas_%A.out
#SBATCH --error=/ceph/home/student.aau.dk/gy53xm/AppliedDL/logs/mas_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=60G
#SBATCH --time=04:00:00

set -euo pipefail

# ── Absolute paths ────────────────────────────────────────────────────────────
PROJECT=/ceph/home/student.aau.dk/gy53xm/AppliedDL
AGENT_DIR=$PROJECT/multi-agent

mkdir -p $PROJECT/logs

# ── Step 1: Merge QLoRA (only needed once) ────────────────────────────────────
if [ ! -d "$PROJECT/qlora-merged" ]; then
    echo "[INFO] Merging QLoRA adapter into base model..."
    cd $PROJECT
    uv run --no-sync python $AGENT_DIR/merge_qlora.py
else
    echo "[INFO] Merged model already exists, skipping merge."
fi
# ── Step 2: Create Mistral chat template (required by vLLM) ──────────────────
cat > $PROJECT/mistral_chat_template.jinja << 'EOF'
{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}
EOF
echo "[INFO] Mistral chat template created."

# ── Step 2: Start Qwen (Advocate + Skeptic) on port 8000 ─────────────────────
echo "[INFO] Starting Qwen server on port 8000..."
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 > $PROJECT/logs/vllm_qwen.log 2>&1 &

QWEN_PID=$!
echo "[INFO] Qwen PID: $QWEN_PID"

# ── Step 3: Start QLoRA Mistral (Judge) on port 8001 ─────────────────────────
echo "[INFO] Starting QLoRA Judge server on port 8001..."
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m vllm.entrypoints.openai.api_server \
    --model $PROJECT/qlora-merged \
    --dtype bfloat16 \
    --chat-template $PROJECT/mistral_chat_template.jinja \
    --served-model-name qlora-mistral-judge \
    --port 8001 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 > $PROJECT/logs/vllm_qlora.log 2>&1 &

QLORA_PID=$!
echo "[INFO] QLoRA PID: $QLORA_PID"

# ── Step 4: Wait for both servers ─────────────────────────────────────────────
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600
    local waited=0
    echo "[INFO] Waiting for $name (port $port)..."
    until curl -s http://localhost:$port/health > /dev/null 2>&1; do
        if [ $waited -ge $max_wait ]; then
            echo "[ERROR] $name did not start within ${max_wait}s"
            kill $QWEN_PID $QLORA_PID 2>/dev/null
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "[INFO] $name is ready!"
}

wait_for_server 8000 "Qwen (Advocate/Skeptic)"
wait_for_server 8001 "QLoRA Mistral (Judge)"

# ── Step 5: Run MAS pipeline ──────────────────────────────────────────────────
echo "[INFO] Starting Multi-Agent debate pipeline..."
cd $AGENT_DIR
uv run --no-sync python -u main.py

echo "[INFO] MAS pipeline complete."

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill $QWEN_PID $QLORA_PID 2>/dev/null || true
echo "[INFO] Servers stopped."
