import json
import time
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import glob
import re
import os



with open('target.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]


# 1. DEFINE THE OUTPUT SCHEMA
# Even though we don't use Pydantic for parsing, we use it to tell vLLM what we want.
class QAPair(BaseModel):
    question: str
    answer: str
    reasoning: str = Field(description="Short explanation of why this is the answer")

# Get the JSON schema to pass to vLLM
json_schema = QAPair.model_json_schema()

# ==========================================
# PART A: BATCH GENERATION (The Candidate Producer)
# ==========================================

def generate_candidates(data: List[str], num_candidates=10) -> List[Dict]:
    """
    Generates N candidates per graphlet using vLLM's guided decoding.
    """
    prompts = []
    metadata = [] # Keep track of which prompt belongs to which graphlet

    prompt_template ="""
You are given a knowledge graph snippet. Generate a question and answer strictly based on the information.



Output format:
{"question": "...", "answer": "...", "reasoning": "..."}
in the question, ask something that can be answered by the graphlet. In the answer, provide a concise response based on the graphlet.
in the reasoning, briefly explain how you arrived at the answer using the graphlet information. Be concise but informative.
Graphlet:
"""

    data = data[:num_candidates]
    # 1. Prepare Batch
    for i, sample in enumerate(data):
        print(sample.keys())
        prompts.append(prompt_template+' '.join(sample['graphlet_text'])  + ' edges info: '.join(str(sample['edges'])))
        metadata.append({"graphlet_id": i, "context": sample['graphlet_text']})

    print(f"--- Generating {len(prompts)} candidates ---")

import pandas as pd

generate_candidates(data, num_candidates=10)