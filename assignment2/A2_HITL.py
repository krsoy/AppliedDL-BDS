#!/usr/bin/env python
# coding: utf-8

# # Part C: LLM → Human HITL Labeling Pipeline
#
# Proper HITL workflow:
#   Step 1 (LLM)  : Model reads claim, outputs suggested label + confidence + rationale
#   Step 2 (Human): Human sees claim AND model reasoning, then either:
#                     [a] accept  → is_green_human = llm_green_suggested
#                     [b] override → is_green_human = opposite, must write a note
#
# This makes the human decision traceable: we always know if they agreed or overrode,
# and why they overrode.
#
# Start vLLM server first:
#   python -m vllm.entrypoints.openai.api_server \
#     --model /ceph/home/student.aau.dk/gy53xm/AppliedDL/qlora-merged \
#     --served-model-name qlora-mistral-judge \
#     --port 8001 --max-model-len 4096 \
#     --gpu-memory-utilization 0.85 --trust-remote-code

import json, re, time, textwrap
from pathlib import Path
from typing import Literal

import requests
import pandas as pd
from pydantic import BaseModel, field_validator, ValidationError

# ── Config ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
CSV_PATH  = ROOT / "hitl_green_100.csv"
MODE      = "interactive"           # "llm_only" | "interactive"

VLLM_URL    = "http://localhost:8001/v1/chat/completions"
MODEL_NAME  = "qlora-mistral-judge"
MAX_TOKENS  = 512
TEMPERATURE = 0.1
TIMEOUT     = 60
BATCH_SIZE  = 10
MAX_RETRIES = 3


# ── Pydantic schema ───────────────────────────────────────────────────────────
class GreenLabel(BaseModel):
    llm_green_suggested: int
    llm_confidence: Literal["low", "medium", "high"]
    llm_rationale: str

    @field_validator("llm_green_suggested")
    @classmethod
    def must_be_binary(cls, v):
        if v not in (0, 1):
            raise ValueError(f"llm_green_suggested must be 0 or 1, got {v}")
        return v

    @field_validator("llm_rationale")
    @classmethod
    def must_be_nonempty(cls, v):
        if not v.strip():
            raise ValueError("llm_rationale must not be empty")
        return v.strip()


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert patent analyst specialising in green / climate-tech patents.
    Your task: decide whether a patent claim describes a GREEN technology.

    Definition of GREEN: the claim must relate to:
      - Renewable energy (solar, wind, hydro, geothermal, tidal, biomass)
      - Energy efficiency improvements
      - Carbon capture, storage, or reduction
      - Sustainable transport or propulsion
      - Waste reduction, recycling, or circular economy
      - Clean water technology
      - Climate adaptation or mitigation

    Rules:
      - Use ONLY the claim text provided. Do NOT use CPC codes or outside knowledge.
      - Be conservative: when genuinely unclear, set confidence to "low".
      - Output ONLY valid JSON — no preamble, no markdown fences, nothing else.

    Required JSON schema (exactly these keys):
    {
      "llm_green_suggested": 0 or 1,
      "llm_confidence": "low" | "medium" | "high",
      "llm_rationale": "1-3 sentences citing specific phrases from the claim."
    }
""").strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def llm_label(text: str) -> GreenLabel | None:
    payload = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Claim text:\n\n{text[:3000]}"},
        ],
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",          "", raw)
            return GreenLabel(**json.loads(raw))
        except ValidationError as e:
            print(f"  [attempt {attempt+1}] Pydantic error: {e}")
        except json.JSONDecodeError as e:
            print(f"  [attempt {attempt+1}] JSON error: {e} | raw={raw[:120]}")
        except Exception as e:
            print(f"  [attempt {attempt+1}] Request error: {e}")
        time.sleep(2)
    return None


def label_to_row(label: GreenLabel | None) -> dict:
    if label is None:
        return {"llm_green_suggested": "", "llm_confidence": "error",
                "llm_rationale": "LLM call failed after all retries."}
    return label.model_dump()


# ── Health check ──────────────────────────────────────────────────────────────
def check_server():
    try:
        r = requests.get("http://localhost:8001/v1/models", timeout=5)
        models = [m["id"] for m in r.json().get("data", [])]
        print(f"vLLM server reachable. Models: {models}\n")
    except Exception as e:
        raise RuntimeError(f"Cannot reach vLLM at {VLLM_URL}\nError: {e}")


# ── Mode: LLM-only (batch, no human) ─────────────────────────────────────────
def run_llm_only(df: pd.DataFrame) -> pd.DataFrame:
    needs_llm = df["llm_confidence"].isna() | (df["llm_confidence"] == "")
    todo      = df[needs_llm].index.tolist()
    print(f"Rows needing LLM annotation: {len(todo)}")

    for i, idx in enumerate(todo):
        doc_id = df.at[idx, "doc_id"]
        text   = str(df.at[idx, "text"])
        print(f"[{i+1}/{len(todo)}] {doc_id} ...", end=" ", flush=True)

        result = label_to_row(llm_label(text))
        df.at[idx, "llm_green_suggested"] = result["llm_green_suggested"]
        df.at[idx, "llm_confidence"]      = result["llm_confidence"]
        df.at[idx, "llm_rationale"]       = result["llm_rationale"]
        print(f"suggested={result['llm_green_suggested']}  conf={result['llm_confidence']}")

        if (i + 1) % BATCH_SIZE == 0 or i == len(todo) - 1:
            df.to_csv(CSV_PATH, index=False)
            print(f"  → checkpoint saved ({i+1} done)")

    return df


# ── Mode: interactive HITL ────────────────────────────────────────────────────
def run_interactive(df: pd.DataFrame) -> pd.DataFrame:
    """
    True HITL loop:
      1. LLM produces suggested label + confidence + rationale
      2. Human is shown the claim text AND the full LLM reasoning
      3. Human explicitly accepts [a] or overrides [o]
         - override requires a mandatory note explaining why
      4. Final label + decision type recorded per row
    """
    needs_human = df["is_green_human"].isna() | (df["is_green_human"] == "")
    todo        = df[needs_human].index.tolist()
    total       = len(todo)
    print(f"Rows to review: {total}\n")

    for i, idx in enumerate(todo):
        text   = str(df.at[idx, "text"])
        doc_id = df.at[idx, "doc_id"]

        # ── Step 1: LLM annotation (skip if already done from a previous run) ──
        llm_done = (
            pd.notna(df.at[idx, "llm_confidence"]) and
            str(df.at[idx, "llm_confidence"]) not in ("", "nan", "error")
        )
        if not llm_done:
            print(f"[{i+1}/{total}] {doc_id} — querying model...", flush=True)
            result = label_to_row(llm_label(text))
            df.at[idx, "llm_green_suggested"] = result["llm_green_suggested"]
            df.at[idx, "llm_confidence"]      = result["llm_confidence"]
            df.at[idx, "llm_rationale"]       = result["llm_rationale"]

        llm_suggested  = df.at[idx, "llm_green_suggested"]
        llm_confidence = df.at[idx, "llm_confidence"]
        llm_rationale  = df.at[idx, "llm_rationale"]

        # ── Step 2: Show human the claim + full LLM reasoning ──────────────────
        print("\n" + "╔" + "═" * 78 + "╗")
        print(f"║  [{i+1}/{total}]  {doc_id:<68}║")
        print("╠" + "═" * 78 + "╣")
        print("║  CLAIM TEXT" + " " * 66 + "║")
        print("╠" + "─" * 78 + "╣")
        for line in textwrap.wrap(text[:1500], width=76):
            print(f"║  {line:<76}║")
        print("╠" + "═" * 78 + "╣")
        print("║  MODEL REASONING" + " " * 61 + "║")
        print("╠" + "─" * 78 + "╣")
        label_str = "GREEN (1)" if str(llm_suggested) == "1" else "NOT GREEN (0)"
        print(f"║  Suggested : {label_str:<65}║")
        print(f"║  Confidence: {str(llm_confidence):<65}║")
        print("║  Rationale :" + " " * 65 + "║")
        for line in textwrap.wrap(str(llm_rationale), width=74):
            print(f"║    {line:<74}║")
        print("╚" + "═" * 78 + "╝")

        # ── Step 3: Human decision ─────────────────────────────────────────────
        print("\n  Actions:")
        print("    [a] Accept model suggestion")
        print("    [o] Override model suggestion  (note required)")
        print("    [s] Skip this row")

        while True:
            action = input("\n  Your choice [a/o/s]: ").strip().lower()
            if action in ("a", "o", "s"):
                break
            print("  Please enter a, o, or s.")

        if action == "s":
            print("  Skipped.\n")
            continue

        if action == "a":
            # Human agrees with LLM
            is_green_human     = int(llm_suggested)
            human_decision_type = "accepted"
            human_notes        = ""

        else:  # action == "o"
            # Human overrides — opposite label, note is mandatory
            is_green_human      = 1 - int(llm_suggested)
            human_decision_type = "overridden"
            print(f"\n  You are overriding the model (changing {llm_suggested} → {is_green_human}).")
            while True:
                human_notes = input("  Reason for override (required): ").strip()
                if human_notes:
                    break
                print("  Note cannot be empty when overriding.")

        # ── Step 4: Record decision ────────────────────────────────────────────
        df.at[idx, "is_green_human"]       = is_green_human
        df.at[idx, "human_decision_type"]  = human_decision_type  # "accepted" | "overridden"
        df.at[idx, "human_notes"]          = human_notes

        decision_icon = "✓" if action == "a" else "✗ OVERRIDE"
        print(f"\n  {decision_icon}  Final label: {is_green_human}  ({i+1}/{total} done)\n")

        df.to_csv(CSV_PATH, index=False)

    return df


# ── Override report ───────────────────────────────────────────────────────────
def report_overrides(df: pd.DataFrame) -> None:
    labelled = df[
        df["is_green_human"].notna() & (df["is_green_human"] != "") &
        df["llm_green_suggested"].notna() & (df["llm_green_suggested"] != "")
    ].copy()
    labelled["is_green_human"]      = labelled["is_green_human"].astype(int)
    labelled["llm_green_suggested"] = labelled["llm_green_suggested"].astype(int)

    total    = len(labelled)
    accepted  = (labelled.get("human_decision_type", "") == "accepted").sum() if "human_decision_type" in labelled else 0
    overrides = (labelled["is_green_human"] != labelled["llm_green_suggested"]).sum()

    print(f"\n{'='*60}")
    print("HITL Summary  (include in README)")
    print(f"{'='*60}")
    print(f"Total reviewed : {total}")
    print(f"Accepted       : {total - overrides}  ({100*(total-overrides)/max(total,1):.1f}%)")
    print(f"Overridden     : {overrides}  ({100*overrides/max(total,1):.1f}%)")

    print("\nExample overrides:")
    overridden_rows = labelled[labelled["is_green_human"] != labelled["llm_green_suggested"]]
    for _, row in overridden_rows.head(3).iterrows():
        print(f"\n  doc_id     : {row['doc_id']}")
        print(f"  Model      : {row['llm_green_suggested']}  (conf: {row['llm_confidence']})")
        print(f"  Human      : {row['is_green_human']}")
        print(f"  Rationale  : {str(row['llm_rationale'])[:200]}")
        print(f"  Human note : {row.get('human_notes', '')}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_server()
    df = pd.read_csv(CSV_PATH, dtype=str)

    # Ensure columns exist
    for col in ["llm_green_suggested", "llm_confidence", "llm_rationale",
                "is_green_human", "human_decision_type", "human_notes"]:
        if col not in df.columns:
            df[col] = ""

    if MODE == "llm_only":
        df = run_llm_only(df)
        df.to_csv(CSV_PATH, index=False)
        print(f"\nAll LLM labels written to {CSV_PATH}")
        print("→ Switch MODE to 'interactive' to do the human review step.")

    elif MODE == "interactive":
        df = run_interactive(df)
        df.to_csv(CSV_PATH, index=False)
        report_overrides(df)

    else:
        raise ValueError(f"Unknown MODE: '{MODE}'. Choose 'llm_only' or 'interactive'.")