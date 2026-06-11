# understand_retrieval.py

from src.ingestion.embedder import get_embeddings
from src.retrieval.vectorstore_manager import load_vectorstore


question = "Why does coffee taste bitter?"

print(f"\nQuestion : {question}")

# STEP 1
print("\nSTEP 1 : Loading Embedding Model")

embeddings = get_embeddings()

# STEP 2
print("\nSTEP 2 : Converting Question Into Vector")

query_vector = embeddings.embed_query(question)

print(f"Vector Length : {len(query_vector)}")

print(f"First 10 Numbers : {query_vector[:10]}")

# STEP 3
print("\nSTEP 3 : Loading FAISS Database")

vectorstore = load_vectorstore()

print("\nSTEP 4 : Manual FAISS Search")

results = vectorstore.similarity_search_with_score(
    question,
    k=5
)

for i, (doc, score) in enumerate(results, start=1):

    print("\n" + "=" * 80)

    print(f"Result : {i}")

    print(f"Similarity Score : {score}")

    print(f"Page : {doc.metadata['page']}")

    print(doc.page_content[:300])




# from src.ingestion.embedder import get_embeddings
# from src.ingestion.loader import load_pdf
# from src.ingestion.chunker import chunk_documents

# question = "Why does coffee taste bitter?"

# embeddings = get_embeddings()

# question_vector = embeddings.embed_query(question)

# docs = load_pdf("data/Coffee-Bible_6439.pdf")
# chunks = chunk_documents(docs)

# print("Checking first 5 chunks manually\n")

# for i in range(5):

#     chunk_text = chunks[i].page_content

#     chunk_vector = embeddings.embed_query(chunk_text)

#     print(f"Chunk {i+1}")
#     print(f"Page : {chunks[i].metadata['page']}")
#     print(f"Vector Length : {len(chunk_vector)}")
#     print("-" * 50)