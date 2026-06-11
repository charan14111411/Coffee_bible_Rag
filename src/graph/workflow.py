from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.graph.state import GraphState
from src.graph.nodes import retrieve_node, grade_documents_node, generate_node

# Initialize the StateGraph with our GraphState
workflow = StateGraph(GraphState)

# Add nodes to the workflow
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)

# Set up edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

# Initialize in-memory checkpointer for conversation memory persistence
checkpointer = MemorySaver()

# Compile the workflow with the checkpointer enabled
app = workflow.compile(checkpointer=checkpointer)
