"""
Configuration settings for the Hate Speech Detection application.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment")

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DOCS_DIR = os.path.join(BASE_DIR, "data", "policy_docs")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")

# Ensure directories exist
os.makedirs(POLICY_DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Classification labels
CLASSIFICATION_LABELS = ["Hate", "Toxic", "Offensive", "Neutral", "Ambiguous"]

# Action mapping
ACTION_MAPPING = {
    "Hate": "Ban user and remove content",
    "Toxic": "Remove content and issue warning",
    "Offensive": "Flag content for review",
    "Neutral": "Allow content",
    "Ambiguous": "Flag for human review"
}

# Model settings
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMPERATURE = 0.1
MAX_TOKENS = 1024

# Retrieval settings
TOP_K_RETRIEVAL = 3
SIMILARITY_THRESHOLD = 0.65