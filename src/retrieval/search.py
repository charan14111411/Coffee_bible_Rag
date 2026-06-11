from src.retrieval.retriever import get_retriever


question = "Why does coffee taste bitter?"

retriever = get_retriever()

docs = retriever.invoke(question)

for i, doc in enumerate(docs, start=1):

    print("=" * 80)

    print(f"Result {i}")

    print("Page:", doc.metadata["page"])

    print()

    print(doc.page_content[:1000])