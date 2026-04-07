import os
from src.utils.logger import get_logger
from src.ingestion.pdf_loader import load_specific_pdfs
from src.ingestion.chunking import chunk_documents
from src.vectorstore.chroma_manager import add_to_vector_store
from src.agents.research_agent import get_rag_chain, get_agent_executor

# Initialize logger for the main entry point
logger = get_logger(__name__)


def process_new_uploads(file_paths: list):
    """
    Orchestrates the ingestion of new files into the existing knowledge base.
    """
    if not file_paths:
        logger.info("No new files provided for upload.")
        return

    logger.info(f"--- Starting Ingestion for {len(file_paths)} files ---")

    # 1. Load PDFs from provided paths
    raw_docs = load_specific_pdfs(file_paths)

    if not raw_docs:
        logger.error("Failed to extract any text from the provided paths.")
        return

    # 2. Chunk the documents
    semantic_chunks = chunk_documents(raw_docs)

    # 3. Add to the existing Vector Store (incremental update)
    add_to_vector_store(semantic_chunks)
    logger.info("Knowledge base successfully updated.")


def run_research_assistant():
    """
    Main loop to simulate an Agentic user session.
    """
    logger.info("--- Agentic Research Assistant Active ---")

    # 1. Initialize the Agent Executor instead of a simple chain
    agent_executor = get_agent_executor()

    print("\n" + "=" * 50)
    print("🤖 INTELLIGENT AGENTIC ASSISTANT (WEEK 3)")
    print("=" * 50)
    print("Type 'exit' to quit or 'upload' to add new files.")

    while True:
        user_input = input("\nEnter your question (or command): ").strip()

        if user_input.lower() == 'exit':
            break

        elif user_input.lower() == 'upload':
            path = input("Enter the full path to the PDF: ").strip()
            if os.path.exists(path):
                process_new_uploads([path])
            else:
                print("Invalid path. Please try again.")
            continue

        # 2. Agentic Query Handling
        try:
            logger.info(f"User Query: {user_input}")

            # IMPORTANT: The AgentExecutor expects the key "input"
            # instead of "question" based on the Hub prompt schema.
            response = agent_executor.invoke({"input": user_input})

            # The response from an AgentExecutor is a dict;
            # the actual text is in the "output" key.
            print(f"\n{response['output']}")

        except Exception as e:
            logger.error(f"Error processing agent query: {e}")
            print("I encountered an error while researching that.")


if __name__ == "__main__":
    run_research_assistant()


if __name__ == "__main__":
    # Optional: Pre-load the sample if the DB is empty
    # You can customize this to automatically index the 'raw' folder on first boot
    run_research_assistant()