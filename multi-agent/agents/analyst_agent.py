from crewai import Agent

def create_analyst_agent(cfg, llm):
    p = cfg['analyst_agent']
    return Agent(name='AnalystAgent', role=p['role'], goal=p['goal'], backstory=p['backstory'], llm=llm)
