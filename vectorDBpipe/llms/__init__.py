"""
vectorDBpipe.llms — LLM client adapters
Supports: OpenAI, Groq, Anthropic, Google, Cohere, Sarvam, DeepSeek, Mistral, and OpenAI-compatible providers.
"""

from .base import BaseLLMClient

__all__ = ["BaseLLMClient"]
