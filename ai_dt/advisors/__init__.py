"""Advisor registry."""

from ai_dt.advisors.combine import combine_atmospheres
from ai_dt.advisors.histogram import HistogramAdvisor
from ai_dt.advisors.ollama_llava import OllamaLLaVAAdvisor

__all__ = [
    "HistogramAdvisor",
    "OllamaLLaVAAdvisor",
    "combine_atmospheres",
]
