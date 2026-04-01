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
    """

    # 1. Initialize the Brain (LLM)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # 2. Initialize the Memory (Retriever)
    vector_db = get_vector_store()
    retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})

    # 3. Design the Prompt (The Instructions)
    template = """
    You are a Senior Research Assistant. Use the provided pieces of retrieved context 
    to answer the user's question. 

    If you don't know the answer based on the context, just say that you don't know. 
    Keep the answer concise and professional.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. The Chain (LCEL)
    # This says:
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

    user_query = "What is this simulation about?"
    print(f"\nUser: {user_query}")

    response = chain.invoke(user_query)
    print(f"\nAI: {response}")