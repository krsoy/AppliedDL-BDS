from crewai import Crew

def run_pipeline(question, agents, tasks):
    crew = Crew(agents=agents, tasks=tasks, verbose=True)
    return crew.kickoff()
