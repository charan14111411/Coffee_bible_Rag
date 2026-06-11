import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
from src.graph.workflow import app as graph

# Set up logging for FastAPI application
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
logger = logging.getLogger("coffee_rag.api")

# Initialize FastAPI application
app = FastAPI(
    title="Coffee Bible RAG API",
    description="FastAPI interface for the LangGraph-based Coffee Bible RAG Assistant",
    version="1.0.0"
)

# Enable CORS for development testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas for request and response validation
class AskRequest(BaseModel):
    question: str
    thread_id: str = "default_thread"

class EvidenceItem(BaseModel):
    page: Any
    start_line: int
    end_line: int

class AskResponse(BaseModel):
    answer: str
    evidence: List[EvidenceItem]


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Endpoint to ask a question to the Coffee Bible RAG assistant.
    The request goes through a LangGraph workflow that retrieves relevant chunks,
    grades their relevance, and generates the final answer using LLM.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    logger.info(f"Received question (Thread: {request.thread_id}): {question}")
    try:
        # Invoke the LangGraph workflow asynchronously with conversation memory checkpointer config
        result = await graph.ainvoke(
            {"question": question},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        logger.info("Answer generated successfully.")
        return {
            "answer": result["answer"],
            "evidence": result["evidence"]
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error invoking graph: {str(e)}")
