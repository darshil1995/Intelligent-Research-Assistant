import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- PROJECT DIRECTORIES ---
# Resolves to the root of your project: Intelligent-Research-Assistant/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# --- MODEL SETTINGS ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# --- RAG PARAMETERS ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
VECTOR_SEARCH_TOP_K = 6

# --- API KEYS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")