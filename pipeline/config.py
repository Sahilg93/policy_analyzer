"""
Centralized Configuration Manager
Loads secure API keys and local folders from environment variables and .env.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger("config")

def load_env(env_path: str = ".env") -> None:
    """Manually parse .env key-value pairs into os.environ to avoid heavy dependencies."""
    p = Path(env_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        os.environ[k] = v
            logger.info(f"Successfully parsed config variables from {p.resolve()}")
        except Exception as e:
            logger.warning(f"Error reading .env config file: {e}")

# Trigger config load on module import
load_env()

# Secure Credential Keys
OPENSTATES_API_KEY = os.getenv("OPENSTATES_API_KEY", "")
BEA_API_KEY = os.getenv("BEA_API_KEY", "")
BLS_API_KEY = os.getenv("BLS_API_KEY", "")
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY", "")

# Host Service Endpoints
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL_EMBED = os.getenv("OLLAMA_MODEL_EMBED", "nomic-embed-text")
OLLAMA_MODEL_GEN = os.getenv("OLLAMA_MODEL_GEN", "llama3")

# Data File Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
HISTORICAL_BILLS_PATH = PROCESSED_DIR / "historical_bills.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "historical_embeddings.parquet"
POLICY_DATASET_PATH = PROCESSED_DIR / "policy_dataset.parquet"
AI_REGISTRY_PATH = PROCESSED_DIR / "ai_classification_registry.parquet"
POLICY_EVENTS_PATH = DATA_DIR / "policy_events.csv"
