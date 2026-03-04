from crewai import Task

def build_retrieval_task(question, agent):
    return Task(description=f"Retrieve evidence for: {question}", expected_output='Evidence snippets grouped by topic', agent=agent)
