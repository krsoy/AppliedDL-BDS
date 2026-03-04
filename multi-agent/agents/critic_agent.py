from crewai import Agent

def create_critic_agent(cfg, llm):
    p = cfg['critic_agent']
    return Agent(name='CriticAgent', role=p['role'], goal=p['goal'], backstory=p['backstory'], llm=llm)
