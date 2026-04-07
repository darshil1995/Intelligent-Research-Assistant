from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper  # NEW IMPORT
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Added OpenAIEmbeddings
from src.utils.logger import get_logger
from src.utils.config import LLM_MODEL

logger = get_logger(__name__)

# 1. Initialize the Judge LLM and Embeddings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=LLM_MODEL))
# Use the same embedding model as your Vector Store for consistency
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# 2. Initialize metrics with BOTH LLM and Embeddings attached
faithfulness = Faithfulness(llm=evaluator_llm)
relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)


async def run_eval_experiment(query: str, response: str, contexts: list):
    """
    Evaluates a live response with an explicitly configured LLM judge.
    """
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=contexts
    )

    logger.info("--- Computing Ragas Scores (Faithfulness & Relevancy) ---")

    # Use the async method appropriate for your 2026 build
    f_score = await faithfulness.single_turn_ascore(sample)
    r_score = await relevancy.single_turn_ascore(sample)

    return {
        "faithfulness": f_score,
        "answer_relevancy": r_score
    }