import json
import logging
import requests
from typing import Generator
from vectorDBpipe.llms.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """
    LLM generation interface for the OpenAI Chat Completions API.
    Supports both blocking (generate_response) and SSE streaming (stream_response).
    """

    def __init__(self, model_name: str, api_key: str, **kwargs):
        if not api_key:
            raise ValueError("OpenAI API Key is required for generation.")

        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"Initialized OpenAI LLM Client for model: {self.model_name}")

    def _build_messages(self, system_prompt: str, user_query: str, retrieved_context: str):
        final_prompt = (
            f"Use the following pieces of retrieved context to answer the question.\n"
            f"If you don't know the answer based on the context, just say that you don't know.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {user_query}\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]

    def generate_response(
        self, system_prompt: str, user_query: str, retrieved_context: str
    ) -> str:
        """Blocking single-call response generation."""
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(system_prompt, user_query, retrieved_context),
            "temperature": 0.3,
        }
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI generate_response failed: {e}")
            raise

    def stream_response(
        self,
        system_prompt: str,
        user_query: str,
        retrieved_context: str,
    ) -> Generator[str, None, None]:
        """
        True SSE token streaming using OpenAI's stream=True parameter.
        Parses `data: {...}` server-sent events and yields delta content tokens.
        Yields an empty string at the end to signal completion.
        """
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(system_prompt, user_query, retrieved_context),
            "temperature": 0.3,
            "stream": True,
        }
        try:
            with requests.post(
                self.url, headers=self.headers, json=payload, stream=True, timeout=60
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception as e:
            logger.error(f"OpenAI stream_response failed: {e}")
            raise
