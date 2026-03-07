from crewai import Agent

def create_synth_agent(cfg, llm):
    p = cfg['synth_agent']
    return Agent(name='SynthAgent', role=p['role'], goal=p['goal'], backstory=p['backstory'], llm=llm)
