from typing import TypedDict, List


class GraphState(TypedDict):
    question: str
    documents: List
    answer: str
    evidence: List
    chat_history: List