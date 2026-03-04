from crewai import Task

def build_synth_task(agent, ctxs, question):
    return Task(description='Integrate evidence, analysis, and critique into final structured JSON.', expected_output='Final structured JSON', agent=agent, context=ctxs)
