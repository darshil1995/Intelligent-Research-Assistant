from langchain_core.messages import SystemMessage
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


def rerank_context(input_data):
    """The 2-Stage Retrieval Logic (Tool Function)."""
    # Check if input is a dict (from Tool) or a string (direct call)
    query = input_data["question"] if isinstance(input_data, dict) else input_data
    vector_db = get_vector_store()
    retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})

    logger.info(f"--- [Stage 1] Retrieving candidates for: {query} ---")
    initial_docs = retriever.invoke(query)

    logger.info("--- [Stage 2] Reranking for Precision ---")
    final_docs = reranker.rerank(query, initial_docs, top_n=5)

    return final_docs


def get_agent_executor():
    """Builds the Multi-Tool Reasoning Agent."""
    logger.info(f"Initializing Agentic System using: {LLM_MODEL}")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)



    client = Client()
    prompt = client.pull_prompt("hwchase17/openai-functions-agent")

    # ZERO-TOLERANCE RESEARCH AUDITOR RULE
    instruction = (
        "You are a Strict Research Verification Bot with ZERO internal knowledge. "
        "Your objective is to provide answers that are 100% grounded in the provided tool results."

        "\n\n--- THE GROUNDING MANIFESTO ---\n"
        "1. If information is NOT present in the 'PDF_Research_Search' or 'tavily_search' outputs, "
        "you MUST state that the information is not available. "
        "2. FORBIDDEN: Do not use your own training data to define, explain, or compare terms. "
        "3. FORBIDDEN: Do not use phrases like 'could refer to', 'generally speaking', or 'in a business context' "
        "unless that exact phrase is found in the tool results."

        "\n\n--- MANDATORY CITATION FORMAT ---\n"
        "4. Every single technical claim or definition MUST end with a citation. "
        "5. Format for PDF: (Source: [filename], Page [X]). "
        "6. Format for Web: (Source: [URL])."

        "\n\n--- HANDLING MISSING DATA ---\n"
        "7. If the user asks for a comparison (e.g., 'how does this differ from business definitions?') "
        "and the business definition is not in your tools, your response must be: "
        "'The provided research documents do not contain a business-specific definition for [Term], "
        "therefore a comparison cannot be made.'"
    )

    # Inject the instruction into the prompt
    prompt.messages.insert(0, SystemMessage(content=instruction))

    # Setup the Toolset
    pdf_tool = get_pdf_search_tool(rerank_context)
    web_tool = get_web_search_tool()
    tools = [pdf_tool, web_tool]

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
        response_dict = agent_executor.invoke(
            {"input": query},
            config=config
        )

        answer = response_dict["output"]

        print("\n" + "=" * 30)
        print("AI ASSISTANT RESPONSE")
        print("=" * 30)
        print(answer)
        print("=" * 30 + "\n")

    except Exception as e:
        logger.error(f"Critical error during agent execution: {e}")