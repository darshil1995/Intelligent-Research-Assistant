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

    # 2. THE UPGRADED PROMPT: Notice the negative constraints and formatting rules
    template = """
    SYSTEM INSTRUCTIONS:
    You are a professional Technical Research Assistant. Your goal is to provide 
    accurate, fact-based answers derived EXCLUSIVELY from the provided context.

    RULES:
    1. Use ONLY the provided context to answer. 
    2. If the answer is not contained within the context, state: "I'm sorry, but the 
       provided documents do not contain information regarding this query."
    3. Do NOT use outside knowledge or "hallucinate" details.
    4. Maintain a formal, objective tone.

    CONTEXT:
    {context}

    USER QUESTION: 
    {question}

    FINAL ANSWER:
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

    user_query = "What is capital of Canada?"
    print(f"\nUser: {user_query}")

    response = chain.invoke(user_query)
    print(f"\nAI: {response}")