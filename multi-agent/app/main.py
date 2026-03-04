import json
from rag.loader import load_pdfs
from rag.vectorstore import build_vectorstore
from rag.retriever import build_retriever
from agents.llm_loader import load_local_llm, load_cloud_llm
from agents.retrieval_agent import create_retrieval_agent
from agents.analyst_agent import create_analyst_agent
from agents.critic_agent import create_critic_agent
from agents.synth_agent import create_synth_agent
from tasks.retrieval_task import build_retrieval_task
from tasks.analysis_task import build_analysis_task
from tasks.critique_task import build_critique_task
from tasks.synth_task import build_synth_task
from core.pipeline import run_pipeline

def load_config():
    with open('config/settings.json', 'r', encoding='utf-8') as f: settings = json.load(f)
    with open('config/prompts.json', 'r', encoding='utf-8') as f: prompts = json.load(f)
    return settings, prompts

def main():
    settings, prompts = load_config()
    question = settings.get('question', 'Synthesize policy for Denmark AI, in the view of Geopolitical Competition between US and China, using all PDFs.')

    docs = load_pdfs(settings['pdf_dir'])
    vs = build_vectorstore(docs, settings['embedding_model'], settings['rag']['chunk_size'], settings['rag']['chunk_overlap'])
    retriever = build_retriever(vs, settings['rag']['top_k'])

    # local_llm = load_local_llm(settings['local_model_path'])
    cloud_llm = load_cloud_llm(settings['cloud_model'])

    r = create_retrieval_agent(prompts, retriever, cloud_llm)
    a = create_analyst_agent(prompts, cloud_llm)
    c = create_critic_agent(prompts, cloud_llm)
    s = create_synth_agent(prompts, cloud_llm)

    t1 = build_retrieval_task(question, r)
    t2 = build_analysis_task(a, t1)
    t3 = build_critique_task(c, t2)
    t4 = build_synth_task(s, [t1, t2, t3], question)

    out = run_pipeline(question, [r, a, c, s], [t1, t2, t3, t4])

    with open('final_output.json', 'w', encoding='utf-8') as f:
        f.write(out)
    print('[INFO] Written final_output.json')

if __name__ == '__main__':
    main()
