from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.utils.config import LLM_MODEL, VECTOR_SEARCH_TOP_K
from src.vectorstore.chroma_manager import get_vector_store


def format_docs(docs):
    """Combine the content of retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    """
        Creates a RAG Chain using LCEL (LangChain Expression Language).
        Logic: Retrieve -> Contextualize -> Generate
        And Upgraded RAG Chain with Strict Guardrails and Temperature Control.
        """
    # 1. Temperature=0 ensures the most deterministic/least 'creative' response
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    vector_db = get_vector_store()
    retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})

    # 2. THE INTEGRATED CHAIN OF THOUGHT PROMPT
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
    # a) Take the question, find context via retriever.
    # b) Pass both to the prompt.
    # c) Pass prompt to LLM.
    # d) Parse the result as a string.
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    # Test the Brain!
    print("---Initializing RAG Chain ---")
    chain = get_rag_chain()

    user_query = "How does the simulation handle capacity planning in Web-based systems?"
    print(f"\nUser: {user_query}")

    response = chain.invoke(user_query)
    print(f"\nAI: {response}")