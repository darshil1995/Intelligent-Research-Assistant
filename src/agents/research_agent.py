from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig, Runnable
from langsmith import Client
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

from src.utils.config import LLM_MODEL, VECTOR_SEARCH_TOP_K
from src.utils.memory_manager import get_session_history
from src.utils.token_counter import validate_context_budget
from src.vectorstore.chroma_manager import get_vector_store
from src.utils.logger import get_logger
from src.utils.reranker import DocumentReranker
from src.tools.web_search_tool import get_web_search_tool
from src.tools.pdf_tool import get_pdf_search_tool

from typing import Any, cast
from langchain_core.runnables import Runnable

# Initialize the reranker once
reranker = DocumentReranker()
logger = get_logger(__name__)


def format_docs(docs):
    """Combines and validates the token budget of retrieved documents."""
    combined_text = "\n\n".join(doc.page_content for doc in docs)
    final_context = validate_context_budget(combined_text, limit=2000)
    return final_context


def rerank_context(input_data):
    """The 2-Stage Retrieval Logic (Tool Function)."""
    query = input_data["question"]
    vector_db = get_vector_store()
    retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})

    logger.info(f"--- [Stage 1] Retrieving candidates for: {query} ---")
    initial_docs = retriever.invoke(query)

    logger.info("--- [Stage 2] Reranking for Precision ---")
    final_docs = reranker.rerank(query, initial_docs, top_n=5)

    return format_docs(final_docs)


def get_agent_executor():
    """Builds the Multi-Tool Reasoning Agent."""
    logger.info(f"Initializing Agentic System using: {LLM_MODEL}")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # 1. Setup the Toolset
    # Pass 'rerank_context' as the function to be used by the tool
    pdf_tool = get_pdf_search_tool(rerank_context)
    web_tool = get_web_search_tool()

    tools = [pdf_tool, web_tool]

    client = Client()
    prompt = client.pull_prompt("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True  # CRITICAL: This captures the RAG output
    )

    return RunnableWithMessageHistory(
        cast(Runnable[Any, Any], executor),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

if __name__ == "__main__":
    logger.info("--- Starting Research Agent Test Run ---")
    try:
        agent_executor = get_agent_executor()
        query = "What is the main objective of the research regarding Web-based systems?"

        # Log BEFORE the long-running process
        logger.info(f"Invoking agent with query: '{query}'")

        # Add a config dictionary for session tracking
        config = RunnableConfig(
            configurable={"session_id": "research_session_1"}
        )
        #config = {"configurable": {"session_id": "research_session_1"}}

        response_dict = agent_executor.invoke(
            {"input": query},
            config=config
        )

        answer = response_dict["output"]
        #print(f"\n🤖 Answer: {answer}")

        print("\n" + "=" * 30)
        print("AI ASSISTANT RESPONSE")
        print("=" * 30)
        print(answer)
        print("=" * 30 + "\n")

    except Exception as e:
        logger.error(f"Critical error during agent execution: {e}")