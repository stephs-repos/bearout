"""Substantiate — fail-closed grounding for RAG.

Every sentence verified against its cited source, or suppressed.
Verifier error rates measured and published.
"""

from substantiate.gate import (
    ClaimVerdict,
    GateOutcome,
    sentence_supported_via_claims,
    validate_with_claims,
)
from substantiate.llm import ChatLLM

__version__ = "0.1.0.dev0"

__all__ = [
    "ChatLLM",
    "ClaimVerdict",
    "GateOutcome",
    "sentence_supported_via_claims",
    "validate_with_claims",
]
