from crewai import Agent
from crewai.tools import BaseTool
from langchain_core.tools.retriever import create_retriever_tool
from pydantic import Field

class RetrieverTool(BaseTool):
    name: str = "document_search"
    description: str = "Search for information in the documents"
    retriever: object = Field(exclude=True)

    def _run(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)

def create_retrieval_agent(cfg, retriever, llm):
    p = cfg['retrieval_agent']
    retriever_tool = RetrieverTool(retriever=retriever)
    return Agent(
        role=p['role'], goal=p['goal'], backstory=p['backstory'], tools=[retriever_tool], llm=llm
    )