import asyncio
import logging
from src.graph.workflow import app as graph

# Set up logging for CLI run to show log statements cleanly
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():

    print("=" * 80)
    print("Coffee Bible RAG Assistant (LangGraph)")
    print("Type 'exit' to quit")
    print("=" * 80)

    while True:

        question = input(
            "\nAsk a Question: "
        ).strip()

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        try:

            # Invoke the LangGraph workflow asynchronously with session config
            result = asyncio.run(
                graph.ainvoke(
                    {
                        "question": question
                    },
                    config={"configurable": {"thread_id": "cli_session"}}
                )
            )

            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)

            print(result["answer"])

            print("\n" + "=" * 80)
            print("EVIDENCE")
            print("=" * 80)

            for i, evidence in enumerate(
                result["evidence"],
                start=1
            ):

                print(
                    f"\nEvidence {i} | Page {evidence['page']} (Lines {evidence['start_line']} - {evidence['end_line']})"
                )

                print("-" * 80)

        except Exception as e:

            print("\nError:")
            print(str(e))


if __name__ == "__main__":
    main()