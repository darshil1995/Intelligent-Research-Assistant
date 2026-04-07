from langchain_core.tools import Tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_pdf_search_tool(rag_function):
    """
    Wraps the local RAG pipeline into a tool.
    Accepts the search function as an argument to avoid circular imports.
    """
    return Tool(
        name="PDF_Research_Search",
        func=lambda q: rag_function({"question": q}),
        description=(
            "USE THIS FIRST for any technical questions regarding Web clusters, "
            "caching protocols, BHR (Byte Hit Ratio), or CWebSim simulation results. "
            "This tool searches the private uploaded research documents."
        )
    )