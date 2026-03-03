from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator


class BaseLLMProvider(ABC):
    """
    Abstract base class for all Large Language Model providers.
    Supports: OpenAI, Gemini, Groq, Anthropic, Cohere, Sarvam, DeepSeek, Mistral.

    All concrete providers MUST implement:
        - __init__(model_name, api_key, **kwargs)
        - generate_response(system_prompt, user_query, retrieved_context) -> str

    Streaming:
        - stream_response() has a default implementation that wraps generate_response()
          into a single-chunk generator, so providers that don't implement native
          streaming still work correctly with stream_query() callers.
        - Override stream_response() in a provider subclass for true token streaming.
    """

    @abstractmethod
    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        Initialize the LLM API client.
        :param model_name: Name of the model (e.g., 'gpt-4o', 'gemini-1.5-flash').
        :param api_key: API key for the provider.
        """
        pass

    @abstractmethod
    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        """
        Generate a full (non-streaming) response.
        :param system_prompt: Instructions defining bot persona and rules.
        :param user_query: The user's question.
        :param retrieved_context: Raw retrieved text from the Vector DB.
        :return: Complete generated string response.
        """
        pass

    def stream_response(
        self,
        system_prompt: str,
        user_query: str,
        retrieved_context: str,
    ) -> Generator[str, None, None]:
        """
        Stream a response token by token.

        Default implementation calls generate_response() and yields the full
        string as a single chunk. Override this in provider subclasses to
        enable true SSE token streaming.

        :param system_prompt: Instructions defining bot persona and rules.
        :param user_query: The user's question.
        :param retrieved_context: Raw retrieved text from the Vector DB.
        :yields: String tokens / chunks.
        """
        yield self.generate_response(system_prompt, user_query, retrieved_context)
