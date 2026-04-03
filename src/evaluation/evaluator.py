from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.config import LLM_MODEL


def evaluate_faithfulness(question, context, answer):
    """
    Using an LLM as a judge to check for hallucinations.
    This is a simplified version of what frameworks like RAGAS do.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    template = """
    You are an expert Grader. Your goal is to check if an AI's answer is FAITHFUL to the provided context.

    Rules:
    1. If the answer contains information NOT present in the context, it is a hallucination.
    2. Give a score from 0 to 10 (10 being perfectly faithful).
    3. Provide a 1-sentence reason for your score.

    Context: {context}
    Answer: {answer}

    Score and Reason:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "answer": answer})


if __name__ == "__main__":
    # Example Test Case
    mock_context = "The company's revenue in 2023 was $50 million."
    mock_answer = "The company made $50 million in 2023 and is planning to double it in 2024."

    print("---Running Faithfulness Evaluation ---")
    result = evaluate_faithfulness("What was the revenue?", mock_context, mock_answer)
    print(f"Evaluation Result: {result}")
    # Note: It should score lower because the '2024' part is not in the context!