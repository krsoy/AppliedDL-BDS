"""
main.py  –  Green Patent Classification via Multi-Agent Debate (CrewAI)
─────────────────────────────────────────────────────────────────────────
Agent 1 (Advocate): Qwen 7B        @ port 8000  → argues FOR green
Agent 2 (Skeptic):  Qwen 7B        @ port 8000  → argues AGAINST green
Agent 3 (Judge):    QLoRA Mistral  @ port 8001  → verdict + rationale
                    called directly via /v1/completions (bypasses CrewAI)
"""

import json
import re
import time
import requests
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task, Crew, Process, LLM


# ── Pydantic output schema ────────────────────────────────────────────────────
class PatentVerdict(BaseModel):
    patent_id:      str   = Field(description="Claim identifier e.g. claim_0")
    claim_text:     str   = Field(description="First 200 chars of the patent claim")
    final_label:    int   = Field(description="1 if green technology, 0 if not")
    confidence:     float = Field(description="Confidence score between 0.0 and 1.0")
    y02_category:   str   = Field(description="Y02 subcategory or none")
    advocate_score: float = Field(description="Advocate confidence score 0.0-1.0")
    skeptic_score:  float = Field(description="Skeptic confidence score 0.0-1.0")
    rationale:      str   = Field(description="2-3 sentence explanation")

    @field_validator("final_label")
    @classmethod
    def label_must_be_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("final_label must be 0 or 1")
        return v

    @field_validator("confidence", "advocate_score", "skeptic_score")
    @classmethod
    def score_must_be_in_range(cls, v):
        return max(0.0, min(1.0, round(v, 3)))

    @field_validator("y02_category")
    @classmethod
    def validate_y02(cls, v):
        valid = {"Y02E", "Y02T", "Y02B", "Y02A", "Y02W", "Y02P", "none", "unknown"}
        return v if v in valid else "unknown"


# ── Y02 context (shared between training and inference) ───────────────────────
Y02_CONTEXT = (
    "Y02 green technology categories:\n"
    "- Y02E: renewable energy (solar, wind, hydro, fuel cells, smart grid)\n"
    "- Y02T: clean transport (electric vehicles, hydrogen, efficient engines)\n"
    "- Y02B: energy efficiency in buildings (insulation, heat pumps, LED)\n"
    "- Y02A: climate change adaptation (flood protection, drought resistance)\n"
    "- Y02W: waste and recycling (circular economy, biofuels from waste)\n"
    "- Y02P: low-carbon production (green chemistry, carbon capture)\n"
)


# ── Config ────────────────────────────────────────────────────────────────────
def load_config():
    with open("config/settings.json", "r") as f:
        settings = json.load(f)
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)
    return settings, prompts


# ── Load LLMs ─────────────────────────────────────────────────────────────────
def load_llms(settings):
    # Advocate + Skeptic: Qwen @ 8000
    qwen_llm = LLM(
        model=f"hosted_vllm/{settings['judge_model_name']}",
        api_base=settings["judge_model_url"],
        api_key="dummy",
        max_tokens=512,
        temperature=0.3,
    )
    return qwen_llm


# ── Select high-risk claims ───────────────────────────────────────────────────
def select_high_risk_claims(settings):
    pool_path = settings["data"]["pool_pseudo_labels_path"]
    n         = settings["data"]["n_high_risk"]
    ucol      = settings["data"]["high_risk_uncertainty_col"]

    pool_df   = pd.read_parquet(pool_path)
    high_risk = (
        pool_df.sort_values(ucol, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    print(f"[INFO] Selected {len(high_risk)} high-risk claims "
          f"(avg uncertainty: {high_risk[ucol].mean():.3f})")
    return high_risk


# ── Build agents (Advocate + Skeptic only, both Qwen) ────────────────────────
def build_agents(prompts, qwen_llm):
    adv_cfg = prompts["advocate_agent"]
    skp_cfg = prompts["skeptic_agent"]

    advocate = Agent(
        role      = adv_cfg["role"],
        goal      = adv_cfg["goal"],
        backstory = adv_cfg["backstory"],
        llm       = qwen_llm,
        verbose   = True,
        max_iter  = 2,
        allow_delegation=False,
    )

    skeptic = Agent(
        role      = skp_cfg["role"],
        goal      = skp_cfg["goal"],
        backstory = skp_cfg["backstory"],
        llm       = qwen_llm,
        verbose   = True,
        max_iter  = 2,
        allow_delegation=False,
    )

    return advocate, skeptic


# ── Build tasks ───────────────────────────────────────────────────────────────
def build_tasks(claim_text, advocate, skeptic, prompts):
    adv_cfg = prompts["advocate_agent"]
    skp_cfg = prompts["skeptic_agent"]

    advocate_task = Task(
        description=(
            f"Analyze this patent claim and argue FOR green (Y02) classification.\n\n"
            f"CLAIM:\n{claim_text}\n\n"
            f"Instructions:\n" + "\n".join(f"- {i}" for i in adv_cfg["instructions"]) +
            f"\n\nCite specific phrases from the claim. "
            f"End your response with: Confidence: X.XX"
        ),
        expected_output=(
            "A structured argument FOR green classification with specific evidence "
            "from the claim text and a confidence score between 0.0 and 1.0."
        ),
        agent=advocate,
    )

    skeptic_task = Task(
        description=(
            f"Analyze this patent claim and argue AGAINST green (Y02) classification.\n\n"
            f"CLAIM:\n{claim_text}\n\n"
            f"Instructions:\n" + "\n".join(f"- {i}" for i in skp_cfg["instructions"]) +
            f"\n\nIdentify greenwashing risks and weaknesses. "
            f"End your response with: Confidence: X.XX"
        ),
        expected_output=(
            "A structured argument AGAINST green classification identifying weaknesses, "
            "greenwashing risks, and a confidence score between 0.0 and 1.0."
        ),
        agent=skeptic,
        context=[advocate_task],
    )

    return advocate_task, skeptic_task


# ── Judge: QLoRA Mistral via /v1/completions ──────────────────────────────────
def mistral_judge(claim_text, claim_idx, advocate_raw, skeptic_raw, settings):
    """
    Call QLoRA Mistral directly via /v1/completions.
    Prompt matches training format exactly:
      System + Y02 context + claim + advocate + skeptic → Verdict + Rationale
    Returns (label, confidence, rationale, y02_category)
    """
    adv_summary = advocate_raw[:400] if advocate_raw else "No argument provided."
    skp_summary = skeptic_raw[:400]  if skeptic_raw  else "No argument provided."

    prompt = (
        "You are a patent classification judge specializing in Y02 green technology.\n\n"
        f"{Y02_CONTEXT}\n"
        f"Patent claim:\n{claim_text[:500]}\n\n"
        f"Advocate argues:\n{adv_summary}\n\n"
        f"Skeptic argues:\n{skp_summary}\n\n"
        "Based on the arguments above, provide your judgment.\n"
        "Verdict:"
    )

    try:
        resp = requests.post(
            f"{settings['advocate_model_url']}/completions",
            json={
                "model":       settings["advocate_model_name"],
                "prompt":      prompt,
                "max_tokens":  200,
                "temperature": 0.1,
                "stop":        ["\n\n", "Patent claim:"],
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["text"].strip()
        print(f"  [Judge] Mistral raw: '{text[:100]}'")
        return parse_mistral_output(text, claim_idx)

    except Exception as e:
        print(f"  [Judge] Mistral call failed: {e}")
        return None, None, f"Mistral call failed: {str(e)[:100]}", "unknown"


def parse_mistral_output(text, claim_idx):
    """
    Parse Mistral output:
      yes
      Rationale: This patent qualifies as Y02E because...
    Returns (label, confidence, rationale, y02_category)
    """
    text_lower = text.lower()

    # Extract verdict from first 30 chars
    label = None
    if text_lower.startswith("yes") or "yes" in text_lower[:30]:
        label = 1
    elif text_lower.startswith("no") or "no" in text_lower[:30]:
        label = 0

    if label is None:
        print(f"  [Judge] claim_{claim_idx}: ambiguous verdict: '{text[:50]}'")
        return None, None, text[:200], "unknown"

    # Extract rationale
    rationale = "No rationale provided."
    if "rationale:" in text_lower:
        rationale = text[text_lower.index("rationale:") + len("rationale:"):].strip()
        rationale = rationale[:300]

    # Extract Y02 category
    y02_category = "none" if label == 0 else "unknown"
    for cat in ["Y02E", "Y02T", "Y02B", "Y02A", "Y02W", "Y02P"]:
        if cat in text:
            y02_category = cat
            break

    return label, 0.85, rationale, y02_category


# ── Run debate for one claim ──────────────────────────────────────────────────
def debate_one_claim(claim_text, claim_idx, advocate, skeptic, prompts, settings):
    advocate_task, skeptic_task = build_tasks(claim_text, advocate, skeptic, prompts)

    crew = Crew(
        agents=[advocate, skeptic],
        tasks=[advocate_task, skeptic_task],
        process=Process.sequential,
        verbose=True,
    )
    crew.kickoff()

    advocate_raw = advocate_task.output.raw if advocate_task.output else ""
    skeptic_raw  = skeptic_task.output.raw  if skeptic_task.output  else ""
    adv_score    = extract_score(advocate_raw)
    skp_score    = extract_score(skeptic_raw)

    # Judge: QLoRA Mistral via /v1/completions
    final_label, confidence, rationale, y02_category = mistral_judge(
        claim_text, claim_idx, advocate_raw, skeptic_raw, settings
    )

    # Fallback if Mistral ambiguous or failed
    if final_label is None:
        print(f"  [WARN] claim_{claim_idx}: Mistral ambiguous, "
              f"falling back to advocate vs skeptic scores")
        final_label  = 1 if adv_score > skp_score else 0
        confidence   = round((adv_score + (1 - skp_score)) / 2, 3)
        rationale    = (f"Fallback: advocate={adv_score:.2f} vs "
                        f"skeptic={skp_score:.2f}. Mistral output ambiguous.")
        y02_category = "unknown"

    return PatentVerdict(
        patent_id      = f"claim_{claim_idx}",
        claim_text     = claim_text[:200],
        final_label    = final_label,
        confidence     = confidence,
        y02_category   = y02_category,
        advocate_score = adv_score,
        skeptic_score  = skp_score,
        rationale      = rationale,
    ).model_dump()


# ── Score extractor ───────────────────────────────────────────────────────────
def extract_score(text):
    if not text:
        return 0.5
    match = re.search(r'[Cc]onfidence[:\s]+([01]\.\d+)', text)
    if match:
        return float(match.group(1))
    matches = re.findall(r'\b(0\.\d{2,}|1\.0)\b', text)
    scores  = [float(m) for m in matches if 0.0 <= float(m) <= 1.0]
    return scores[-1] if scores else 0.5


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    settings, prompts = load_config()

    qwen_llm = load_llms(settings)
    print(f"[INFO] Advocate/Skeptic: {settings['judge_model_name']} "
          f"@ {settings['judge_model_url']}")
    print(f"[INFO] Judge (QLoRA):    {settings['advocate_model_name']} "
          f"@ {settings['advocate_model_url']}")

    high_risk_df = select_high_risk_claims(settings)

    results = []
    total   = len(high_risk_df)

    for i, (_, row) in enumerate(high_risk_df.iterrows()):
        claim_text = str(row["text"])[:2000]
        print(f"\n[{i+1}/{total}] Debating claim {i}...")
        print(f"  Preview: {claim_text[:80]}...")

        # Rebuild agents every claim to avoid context accumulation
        advocate, skeptic = build_agents(prompts, qwen_llm)

        try:
            result = debate_one_claim(
                claim_text = claim_text,
                claim_idx  = i,
                advocate   = advocate,
                skeptic    = skeptic,
                prompts    = prompts,
                settings   = settings,
            )
            result["true_label_lr"]  = int(row.get("pseudo_label_lr", -1))
            result["uncertainty_lr"] = float(row.get("uncertainty_lr", -1))
            results.append(result)

            print(f"  → label={result['final_label']}  "
                  f"conf={result['confidence']}  "
                  f"y02={result['y02_category']}")
            print(f"  → rationale: {result['rationale'][:100]}")

        except Exception as e:
            print(f"  [ERROR] claim_{i}: {e}")
            err_record = PatentVerdict(
                patent_id      = f"claim_{i}",
                claim_text     = claim_text[:200],
                final_label    = 0,
                confidence     = 0.0,
                y02_category   = "unknown",
                advocate_score = 0.0,
                skeptic_score  = 0.0,
                rationale      = f"Pipeline error: {str(e)[:200]}"
            ).model_dump()
            err_record["true_label_lr"]  = int(row.get("pseudo_label_lr", -1))
            err_record["uncertainty_lr"] = float(row.get("uncertainty_lr", -1))
            results.append(err_record)

        time.sleep(0.3)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_path = settings["output"]["results_path"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved {len(results)} results → {out_path}")

    summary_df   = pd.DataFrame(results)
    summary_path = settings["output"]["summary_path"]
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary CSV → {summary_path}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    valid = summary_df[summary_df["final_label"].isin([0, 1])]
    print(f"\n=== MAS Results ===")
    print(f"Total:          {len(results)}")
    print(f"green=1:        {(valid['final_label']==1).sum()}")
    print(f"green=0:        {(valid['final_label']==0).sum()}")
    if len(valid) > 0:
        print(f"Avg confidence: {valid['confidence'].mean():.3f}")
        agree = (valid["final_label"] == valid["true_label_lr"]).mean()
        print(f"Agreement w/LR: {agree:.3f}")


if __name__ == "__main__":
    main()