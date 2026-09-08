import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tests must never depend on the developer's .env, and must never make a real
# Groq or Qdrant call.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QUERY_EXPANSION_MODE", "lexicon")
