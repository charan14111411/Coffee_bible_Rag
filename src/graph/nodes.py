import asyncio
import logging
from src.retrieval.retriever import get_retriever
from src.llm.groq_client import get_llm

logger = logging.getLogger("coffee_rag.nodes")

# Initialize LLM instance (Groq client initialization is fast)
llm = get_llm()


async def invoke_with_retry(prompt, max_attempts=3, delay=2):
    for attempt in range(max_attempts):
        try:
            return await llm.ainvoke(prompt)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            logger.warning(f"LLM invoke failed (attempt {attempt+1}/{max_attempts}): {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)


async def retrieve_node(state):
    """
    Retrieve documents from the vector store based on the question (rephrased using history if available).
    """
    logger.info("--- RETRIEVE NODE ---")
    question = state["question"]
    chat_history = state.get("chat_history") or []
    
    # If conversation history exists, rephrase the follow-up question to be standalone
    if chat_history:
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
        rephrase_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone search query. Do NOT answer the question. Just return the rephrased standalone query in English.

Conversation History:
{history_str}

Follow-up Question: {question}

Standalone Query:"""
        response = await invoke_with_retry(rephrase_prompt)
        search_query = response.content.strip()
        logger.info(f"Rephrased follow-up question '{question}' to standalone search query: '{search_query}'")
    else:
        search_query = question
        logger.info(f"First message in thread. Running direct query: '{search_query}'")
        
    # Lazily fetch the retriever to load the embedding model only when needed
    retriever = get_retriever()
    documents = await retriever.ainvoke(search_query)
    logger.info(f"Retrieved {len(documents)} document chunks.")
    return {"documents": documents}


async def grade_documents_node(state):
    """
    Filter retrieved documents to keep only those relevant to the question in parallel.
    """
    logger.info("--- GRADE DOCUMENTS NODE (RELEVANCE CHECK) ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        logger.info("No documents to grade.")
        return {"documents": []}
        
    filtered_documents = []
    
    # Prompt for relevance checking
    relevance_prompt = """You are an expert grading assistant.
Determine if the following retrieved document context is relevant to the user's question.

Retrieved Document Context:
{context}

User Question:
{question}

Answer with exactly one word: 'yes' if the document has any relevant information to the question, or 'no' if it is completely irrelevant. Do not include any other text or explanation."""

    # Limit concurrent calls to Groq to prevent hitting rate limits
    sem = asyncio.Semaphore(2)
    
    async def grade_doc(doc):
        async with sem:
            prompt = relevance_prompt.format(context=doc.page_content, question=question)
            return await invoke_with_retry(prompt)

    # Construct prompts and schedule them concurrently with rate limiting
    tasks = [grade_doc(doc) for doc in documents]
        
    logger.info(f"Grading {len(documents)} documents with concurrency limit...")
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, (doc, response) in enumerate(zip(documents, responses), start=1):
        page = doc.metadata.get("page", "Unknown")
        
        # Gracefully handle API rate limits or connection failures by rejecting
        if isinstance(response, Exception):
            logger.error(f"  -> Error grading Document {i} (Page {page}): {response}. Defaulting to IRRELEVANT.")
            continue
            
        grade = response.content.strip().lower()
        
        if "yes" in grade:
            logger.info(f"  -> Document {i} (Page {page}): RELEVANT")
            filtered_documents.append(doc)
        else:
            logger.info(f"  -> Document {i} (Page {page}): IRRELEVANT (Filtered)")
            
    logger.info(f"Filtered down to {len(filtered_documents)} / {len(documents)} relevant documents.")
    return {"documents": filtered_documents}


async def generate_node(state):
    """
    Generate an answer based on relevant documents, chat history, and question.
    """
    logger.info("--- GENERATE NODE ---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history") or []
    
    if not documents:
        answer = "I'm sorry, but I couldn't find any relevant information in the Coffee Bible documents to answer your question."
        new_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        return {"answer": answer, "evidence": [], "chat_history": new_history}
        
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Format chat history for RAG prompt
    if chat_history:
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history])
    else:
        history_str = "No conversation history."
        
    RAG_WITH_HISTORY_PROMPT = """You are a Coffee Bible Assistant.

Use the provided context and conversation history to answer the follow-up question.

Rules:
1. Answer using information from the context.
2. If the context contains partial information, provide the best possible answer.
3. Do NOT say "I could not find the answer" unless the context is completely unrelated.
4. Be concise.
5. Mention relevant details from the context.

Context:
{context}

Conversation History:
{chat_history}

Follow-up Question:
{question}

Answer:"""

    prompt = RAG_WITH_HISTORY_PROMPT.format(
        context=context,
        chat_history=history_str,
        question=question
    )
    
    response = await invoke_with_retry(prompt)
    answer = response.content
    
    new_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    evidence = []
    for doc in documents:
        evidence.append({
            "page": doc.metadata.get("page", "Unknown"),
            "start_line": doc.metadata.get("start_line", 1),
            "end_line": doc.metadata.get("end_line", 1)
        })
        
    return {"answer": answer, "evidence": evidence, "chat_history": new_history}
