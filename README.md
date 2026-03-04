# CrewAI + RAG (Multi-PDF, Multi-Agent Debate)

**Title (from config):** AI Governance & Ethics – Aalborg University (Denmark Policy Brief)

## Quick Start
```bash
python3 -m venv crew_env
source crew_env/bin/activate
pip install -r requirements.txt
python app/main.py
```

## Run on SLURM/Cluster
```bash
sbatch run.sbatch
```

## Configure
- Edit `config/settings.json` for paths/models.
- Edit `config/prompts.json` to change agent roles, instructions, debate rounds, and topic outline.
