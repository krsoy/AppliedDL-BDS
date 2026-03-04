"""
main.py  –  Green Patent Classification via Multi-Agent Debate (CrewAI)
─────────────────────────────────────────────────────────────────────────
Agent 1 (Advocate): Qwen 14B        @ port 8000  → argues FOR green
Agent 2 (Skeptic):  Qwen 14B        @ port 8000  → argues AGAINST green
Agent 3 (Judge):    QLoRA Mistral   @ port 8001  → final JSON verdict
                    output enforced by Pydantic
"""

import json
import re
import time
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field, field_validator

from crewai import Agent, Task, Crew, Process, LLM


# ── Pydantic output schema for Judge ─────────────────────────────────────────
class PatentVerdict(BaseModel):
    patent_id:      str   = Field(description="Claim identifier e.g. claim_0")
    claim_text:     str   = Field(description="First 200 chars of the patent claim")
    final_label:    int   = Field(description="1 if green technology, 0 if not")
    confidence:     float = Field(description="Confidence score between 0.0 and 1.0")
    y02_category:   str   = Field(description="Y02 subcategory: Y02E/Y02T/Y02B/Y02A/Y02W/Y02P or none")
    advocate_score: float = Field(description="Advocate's confidence score 0.0-1.0")
    skeptic_score:  float = Field(description="Skeptic's confidence score 0.0-1.0")
    rationale:      str   = Field(description="2-3 sentence explanation of the decision")

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


# ── Config ────────────────────────────────────────────────────────────────────
def load_config():
    with open("config/settings.json", "r") as f:
        settings = json.load(f)
    with open("config/prompts.json", "r") as f:
        prompts = json.load(f)
    return settings, prompts


# ── Load LLMs via CrewAI ──────────────────────────────────────────────────────
def load_llms(settings):
    qwen_llm = LLM(
        model=f"hosted_vllm/{settings['advocate_model_name']}",
        api_base=settings["advocate_model_url"],
        api_key="dummy",
        # max_tokens=512,
        # temperature=0.3,
    )

    # Judge: QLoRA fine-tuned Mistral @ port 8001
    judge_llm = LLM(
        model=f"hosted_vllm/{settings['judge_model_name']}",
        api_base=settings["judge_model_url"],
        api_key="dummy",
        # max_tokens=512,
        # temperature=0.1,
    )

    return qwen_llm, judge_llm


# ── Select 100 high-risk claims ───────────────────────────────────────────────
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


# ── Build agents ──────────────────────────────────────────────────────────────
def build_agents(prompts, qwen_llm, judge_llm):
    adv_cfg = prompts["advocate_agent"]
    skp_cfg = prompts["skeptic_agent"]
    jdg_cfg = prompts["judge_agent"]

    advocate = Agent(
        role      = adv_cfg["role"],
        goal      = adv_cfg["goal"],
        backstory = adv_cfg["backstory"],
        llm       = qwen_llm,
        verbose   = False,
        max_iter=2,  # 强制最多只允许 2 轮思考/重试
        allow_delegation=False,  # 严禁它把任务转交给别人
    )

    skeptic = Agent(
        role      = skp_cfg["role"],
        goal      = skp_cfg["goal"],
        backstory = skp_cfg["backstory"],
        llm       = qwen_llm,
        verbose   = False,
        max_iter=2,  # 强制最多只允许 2 轮思考/重试
        allow_delegation=False,  # 严禁它把任务转交给别人
    )

    judge = Agent(
        role      = jdg_cfg["role"],
        goal      = jdg_cfg["goal"],
        backstory = jdg_cfg["backstory"],
        llm       = judge_llm,
        verbose   = False,
        max_iter=2,  # 强制最多只允许 2 轮思考/重试
        allow_delegation=False,  # 严禁它把任务转交给别人

    )

    return advocate, skeptic, judge


# ── Build tasks for one claim ─────────────────────────────────────────────────
def build_tasks(claim_text, claim_idx, advocate, skeptic, judge, prompts):
    adv_cfg = prompts["advocate_agent"]
    skp_cfg = prompts["skeptic_agent"]
    jdg_cfg = prompts["judge_agent"]

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

    judge_task = Task(
        description=(
            f"You have received arguments from the Advocate and Skeptic "
            f"about this patent claim.\n\n"
            f"PATENT CLAIM:\n{claim_text}\n\n"
            f"Instructions:\n" + "\n".join(f"- {i}" for i in jdg_cfg["instructions"]) +
            f"\n\nUse patent_id: claim_{claim_idx}\n"
            f"You MUST return a valid JSON matching the required schema."
        ),
        expected_output=(
            "A valid JSON object with fields: patent_id, claim_text, "
            "final_label (0 or 1), confidence (0.0-1.0), y02_category, "
            "advocate_score, skeptic_score, rationale."
        ),
        output_pydantic=PatentVerdict,   # ← CrewAI enforces Pydantic schema
        agent=judge,
        context=[advocate_task, skeptic_task],
    )

    return advocate_task, skeptic_task, judge_task


# ── Run debate for one claim ──────────────────────────────────────────────────
def debate_one_claim(claim_text, claim_idx, advocate, skeptic, judge, prompts):
    advocate_task, skeptic_task, judge_task = build_tasks(
        claim_text, claim_idx, advocate, skeptic, judge, prompts
    )

    crew = Crew(
        agents=[advocate, skeptic, judge],
        tasks=[advocate_task, skeptic_task, judge_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    # --- 插入 Debug 代码开始 ---
    # 检查 Pydantic 是否成功解析
    is_pydantic_ok = hasattr(result, "pydantic") and isinstance(result.pydantic, PatentVerdict)

    if not is_pydantic_ok:
        print(f"\n[DEBUG] Claim_{claim_idx} Pydantic 解析失败！")
        # 打印 Judge 实际返回的原始文本，看看是格式错误还是根本没返回 JSON
        raw_output = result.raw if hasattr(result, "raw") else str(result)
        print(f"[DEBUG] Raw Judge Output (first 300 chars): {raw_output[:300]}...")
    # --- 插入 Debug 代码结束 ---

    # Extract individual task outputs for fallback
    advocate_output = advocate_task.output.raw if advocate_task.output else ""
    skeptic_output = skeptic_task.output.raw if skeptic_task.output else ""

    if is_pydantic_ok:
        return result.pydantic.model_dump()

    # Pydantic failed → fallback with real extracted scores
    judge_raw = result.raw if hasattr(result, "raw") else str(result)
    return parse_judge_output(
        judge_raw, claim_text, claim_idx,
        advocate_output, skeptic_output
    )

# ── Fallback parser (when Pydantic validation fails) ─────────────────────────
def parse_judge_output(raw, claim_text, idx, advocate_raw="", skeptic_raw=""):
    # Try direct JSON parse + Pydantic validation
    try:
        data = json.loads(raw)
        return PatentVerdict(**data).model_dump()
    except Exception:
        pass

    # Try extracting JSON block from surrounding text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return PatentVerdict(**data).model_dump()
        except Exception:
            pass

    # Final fallback: extract real scores from agent outputs
    adv_score  = extract_score(advocate_raw)
    skp_score  = extract_score(skeptic_raw)
    label      = 1 if adv_score > skp_score else 0
    confidence = round((adv_score + (1 - skp_score)) / 2, 3)

    print(f"  [WARN] claim_{idx}: Pydantic + JSON parse failed, "
          f"deriving from scores (adv={adv_score:.2f}, skp={skp_score:.2f})")

    # Build a valid PatentVerdict via Pydantic even in fallback
    return PatentVerdict(
        patent_id      = f"claim_{idx}",
        claim_text     = claim_text[:200],
        final_label    = label,
        confidence     = confidence,
        y02_category   = "unknown",
        advocate_score = adv_score,
        skeptic_score  = skp_score,
        rationale      = (
            f"Pydantic/JSON parse failed. Label derived from "
            f"advocate ({adv_score:.2f}) vs skeptic ({skp_score:.2f}) scores. "
            f"Raw: {str(raw)[:200]}"
        )
    ).model_dump()


def extract_score(text):
    """Extract confidence score from agent response text."""
    if not text:
        return 0.5
    # Explicit pattern: "Confidence: 0.85"
    match = re.search(r'[Cc]onfidence[:\s]+([01]\.\d+)', text)
    if match:
        return float(match.group(1))
    # Fallback: last decimal in [0,1]
    matches = re.findall(r'\b(0\.\d{2,}|1\.0)\b', text)
    scores  = [float(m) for m in matches if 0.0 <= float(m) <= 1.0]
    return scores[-1] if scores else 0.5


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    settings, prompts = load_config()

    qwen_llm, judge_llm = load_llms(settings)
    print(f"[INFO] Advocate/Skeptic: {settings['advocate_model_name']} "
          f"@ {settings['advocate_model_url']}")
    print(f"[INFO] Judge (QLoRA):    {settings['judge_model_name']} "
          f"@ {settings['judge_model_url']}")

    advocate, skeptic, judge = build_agents(prompts, qwen_llm, judge_llm)
    high_risk_df = select_high_risk_claims(settings)

    results = []
    total   = len(high_risk_df)

    for i, (_, row) in enumerate(high_risk_df.iterrows()):
        claim_text = str(row["text"])
        # limit text size to 15000
        claim_text = claim_text[:8000 if len(claim_text) > 8000 else len(claim_text)]
        print(f"\n[{i+1}/{total}] Debating claim {i}...")
        print(f"  Preview: {claim_text[:80]}...")

        try:
            result = debate_one_claim(
                claim_text = claim_text,
                claim_idx  = i,
                advocate   = advocate,
                skeptic    = skeptic,
                judge      = judge,
                prompts    = prompts,
            )
            result["true_label_lr"]  = int(row.get("pseudo_label_lr", -1))
            result["uncertainty_lr"] = float(row.get("uncertainty_lr", -1))
            results.append(result)

            print(f"  → label={result['final_label']}  "
                  f"conf={result.get('confidence', '?')}  "
                  f"y02={result.get('y02_category', '?')}")

        except Exception as e:
            print(f"  [ERROR] claim_{i}: {e}")
            # Even errors go through Pydantic for consistent schema
            results.append(PatentVerdict(
                patent_id      = f"claim_{i}",
                claim_text     = claim_text[:200],
                final_label    = 0,
                confidence     = 0.0,
                y02_category   = "unknown",
                advocate_score = 0.0,
                skeptic_score  = 0.0,
                rationale      = f"Pipeline error: {str(e)[:200]}"
            ).model_dump())

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