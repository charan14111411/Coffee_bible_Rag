from src.retrieval.retriever import get_retriever
from src.llm.groq_client import get_llm
from src.llm.prompts import RAG_PROMPT
retriever = get_retriever()
llm = get_llm()
def ask_question(question: str):

    # Load Retriever
    retriever = get_retriever()

    # Retrieve Top Chunks
    docs = retriever.invoke(question)

    # Build Context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create Prompt
    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    # Load LLM
    llm = get_llm()
    print("\n" + "=" * 80)
    print("RETRIEVED CONTEXT")
    print("=" * 80)

    print(context[:3000])
    # Generate Answer
    response = llm.invoke(prompt)

    evidence = []

    for doc in docs:

        evidence.append(
            {
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content
            }
        )   
    return {
        "question": question,
        "answer": response.content,
        "evidence": evidence,
    }