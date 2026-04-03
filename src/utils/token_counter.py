import tiktoken
from src.utils.config import LLM_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def count_tokens(text: str, model: str = LLM_MODEL) -> int:
    """
    Counts the number of tokens in a string for a specific model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for newer models
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = len(encoding.encode(text))
    return num_tokens


def validate_context_budget(context: str, limit: int = 3000):
    """
    Ensures the retrieved context doesn't exceed a safe token limit.
    """
    tokens = count_tokens(context)
    if tokens > limit:
        logger.warning(f"Context exceeds budget ({tokens}/{limit} tokens). Truncating...")
        # Simple truncation logic (Senior note: In production, use smarter pruning)
        return context[:limit * 4]  # Rough character estimate

    logger.info(f"Context within budget: {tokens} tokens.")
    return context