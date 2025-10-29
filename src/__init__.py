"""
AI Health Information Assistant
A production-ready healthcare NLP system for medical Q&A
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easier access
from .config import settings
from .intent_classifier import IntentClassifier
from .qa_engine import QAEngine
from .external_apis import ExternalAPIManager

__all__ = [
    "settings",
    "IntentClassifier",
    "QAEngine",
    "ExternalAPIManager",
]