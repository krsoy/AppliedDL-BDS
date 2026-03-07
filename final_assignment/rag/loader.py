import glob
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(pdf_dir):
    docs = []
    for p in glob.glob(f"{pdf_dir}/*.pdf"):
        docs.extend(PyPDFLoader(p).load())
    return docs
