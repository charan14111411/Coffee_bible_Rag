from src.ingestion.embedder import get_embeddings
from langchain_community.vectorstores import FAISS

_vectorstore = None


def load_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore