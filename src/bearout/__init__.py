"""Bearout — fail-closed grounding for RAG.

Every sentence verified against its cited source, or suppressed.
The verifier's own error rates measured and published.
"""

from bearout.gate import (
    GateOutcome,
    split_sentences,
    validate_grounding,
    verify_sentence,
)
from bearout.llm import ChatLLM

__version__ = "0.0.2"

__all__ = [
    "ChatLLM",
    "GateOutcome",
    "split_sentences",
    "validate_grounding",
    "verify_sentence",
]
