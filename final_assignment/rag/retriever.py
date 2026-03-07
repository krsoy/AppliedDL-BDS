def build_retriever(vs, top_k):
    return vs.as_retriever(k=top_k)
