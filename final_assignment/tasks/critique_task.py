from crewai import Task

def build_critique_task(agent, ctx):
    return Task(description='Critique the analysis: identify weaknesses, alternatives, and gaps; suggest retrieval queries if needed.', expected_output='Critique list', agent=agent, context=[ctx])
