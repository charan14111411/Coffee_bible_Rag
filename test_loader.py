from src.llm.groq_client import get_llm

llm = get_llm()

response = llm.invoke(
    "What is coffee?"
)

print(response.content)