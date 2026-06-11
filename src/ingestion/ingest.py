from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import get_embeddings

from langchain_community.vectorstores import FAISS


documents = load_pdf(
    "data/Coffee-Bible_6439.pdf"
)

chunks = chunk_documents(documents)

embeddings = get_embeddings()

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local(
    "vectorstore/faiss_index"
)

print("FAISS index saved successfully")