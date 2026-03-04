from crewai import Task

def build_analysis_task(agent, ctx):
    return Task(description='Analyze retrieved evidence and build structured arguments for Denmark, what should Denmark do between US and China AI competition, and what role Denmark should play.', expected_output='Structured analysis', agent=agent, context=[ctx])
