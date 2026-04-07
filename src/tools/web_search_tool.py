import os
from langchain_tavily import TavilySearch  # The new class

from src.utils.config import VECTOR_SEARCH_TOP_K
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_web_search_tool():
    """
    Returns the official TavilySearch tool.
    """
    if not os.getenv("TAVILY_API_KEY"):
        logger.error("TAVILY_API_KEY missing from environment.")
        return None

    logger.info("Initializing high-performance TavilySearch tool.")

    # TavilySearch uses 'max_results' (equivalent to 'k')
    return TavilySearch(max_results=VECTOR_SEARCH_TOP_K, topic="general")