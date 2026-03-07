from crewai import LLM

def load_local_llm(path):
    return LLM(
        model=f"hosted_vllm/{path}",
        api_base="http://localhost:8000/v1",
        api_key="dummy",
        max_tokens=1024,
    )

def load_cloud_llm(name):
    return LLM(model=f"openai/{name}")