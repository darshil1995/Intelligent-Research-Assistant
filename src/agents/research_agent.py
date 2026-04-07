from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.utils.config import LLM_MODEL, VECTOR_SEARCH_TOP_K
from src.utils.token_counter import validate_context_budget
from src.vectorstore.chroma_manager import get_vector_store
from src.utils.logger import get_logger
from src.utils.reranker import DocumentReranker

# Initialize the reranker once
reranker = DocumentReranker()

logger = get_logger(__name__)

def format_docs(docs):
    """
        Combine the content of retrieved documents into a single string
        and validate the token budget of retrieved documents.
    """

    for i, doc in enumerate(docs):
        logger.info(f"Chunk {i + 1} content: {doc.page_content[:100]}...")  # See the first 100 chars

    logger.info(f"Retriever found {len(docs)} relevant chunks from the Vector Store.")
    combined_text = "\n\n".join(doc.page_content for doc in docs)
    # Budgeting the context before sending to LLM
    final_context = validate_context_budget(combined_text, limit=2000)
    logger.info(f"Final context formatted and validated.")

    return final_context


# ... (imports stay the same)

# 1. Update the function signature to only take input_data
def rerank_context(input_data):
    query = input_data["question"]

    vector_db = get_vector_store()
    # High Recall: fetch 15 candidates
    retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})

    logger.info(f"--- [Stage 1] Retrieving top 15 candidates for: {query} ---")
    initial_docs = retriever.invoke(query)

    # Log the first 3 raw matches to see if 'References' are present
    for i, doc in enumerate(initial_docs[:3]):
        logger.info(f"Raw Match {i + 1}: {doc.page_content[:50]}...")

    # 2. Stage 2: Reranking (Precision)
    logger.info("--- [Stage 2] Reranking candidates for relevance ---")
    final_docs = reranker.rerank(query, initial_docs, top_n=5)

    # Log the new top 3 to see if the order changed
    for i, doc in enumerate(final_docs[:3]):
        logger.info(
            f"Reranked Match {i + 1} (Score: {doc.metadata.get('rerank_score', 0):.4f}): {doc.page_content[:50]}...")

    return format_docs(final_docs)


def get_rag_chain():
    logger.info(f"Initializing RAG Chain using model: {LLM_MODEL}")

    # Notice we don't need to define retriever here anymore
    # since rerank_context handles its own retrieval stage.
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    template = """
            SYSTEM INSTRUCTIONS:
            You are a Senior Technical Research Assistant. Your goal is to provide 
            accurate, fact-based answers derived EXCLUSIVELY from the provided context.
            You must follow a structured reasoning process to eliminate hallucinations.

            RULES:
            1. Use ONLY the provided context to answer. Do NOT use outside knowledge.
            2. If the answer is not in the context, state: "I'm sorry, but the provided 
               documents do not contain information regarding this query."
            3. Maintain a formal, objective, and professional tone.
            4. You must provide a 'THOUGHT' section followed by a 'FINAL ANSWER' section.

            YOUR PROCESS:
            1. THOUGHT: Analyze the user's question and identify the specific facts 
               needed from the context. Note which parts of the retrieved data are relevant.
            2. FINAL ANSWER: Provide the concise answer based ONLY on the thought process 
               and retrieved context above.

            CONTEXT:
            {context}

            USER QUESTION: 
            {question}

            YOUR RESPONSE (Following the THOUGHT/FINAL ANSWER format):
            """

    prompt = ChatPromptTemplate.from_template(template)

    # 3. The Chain (LCEL)
    rag_chain = (
        # input_data is passed automatically to rerank_context
            RunnablePassthrough.assign(
                context=lambda x: rerank_context(x)
            )
         #   {"context": rerank_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    logger.info("RAG Chain successfully constructed.")
    return rag_chain

if __name__ == "__main__":
    # Test the Brain!
    logger.info("--- Starting Research Agent Test Run ---")
    try:
        chain = get_rag_chain()
        query = "What is the main objective of the research regarding Web-based systems?"

        logger.info(f"Invoking chain with query: '{query}'")

        # CHANGE THIS LINE TOO:
        response = chain.invoke({"question": query})

        print("\n" + "=" * 30)
        print("AI ASSISTANT RESPONSE")
        print("=" * 30)
        print(response)
        print("=" * 30 + "\n")

    except Exception as e:
        logger.error(f"Critical error during agent execution: {e}")