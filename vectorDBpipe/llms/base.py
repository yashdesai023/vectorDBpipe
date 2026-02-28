from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLLMProvider(ABC):
    """
    Abstract base class for all Large Language Model providers (e.g., OpenAI, Gemini, Sarvam).
    """

    @abstractmethod
    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        Initialize the LLM API client.
        :param model_name: Name of the model (e.g., 'gpt-4o', 'gemini-1.5-flash').
        :param api_key: Required API Key for the provider.
        """
        pass

    @abstractmethod
    def generate_response(self, system_prompt: str, user_query: str, retrieved_context: str) -> str:
        """
        Generate a response based on the search context using Retrieval Augmented Generation.
        :param system_prompt: Instructions defining the bot persona and rules.
        :param user_query: The actual question asked by the user.
        :param retrieved_context: The raw text knowledge retrieved from the Vector Database.
        :return: The generated string response.
        """
        pass
