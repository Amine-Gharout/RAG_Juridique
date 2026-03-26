"""Legal RAG package built around LangGraph and Groq."""

from .config import Settings, load_settings
from .graph import LegalRAGApp

__all__ = ["Settings", "load_settings", "LegalRAGApp"]
