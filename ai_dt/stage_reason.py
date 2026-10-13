"""Backward-compatible import. The text reasoner lives in ``ai_dt.reasoner``."""

from ai_dt.reasoner import propose_stages, skipped, validate_proposal

__all__ = ["propose_stages", "skipped", "validate_proposal"]
