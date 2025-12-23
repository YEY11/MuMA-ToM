"""
Large Language Model (LLM) Client
Unified interface for text generation with different backends
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from LIMP_Poker_V3.config import config

from .base import BaseModelClient


class LLMClient(BaseModelClient):
    """
    Unified LLM client supporting multiple backends through OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ):
        super().__init__(max_retries=max_retries)

        self.api_key = api_key or config.API_KEY or config.LLM_API_KEY
        self.base_url = base_url or config.API_BASE_URL or config.LLM_BASE_URL
        self.model = model or config.LLM_MODEL_NAME

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        logger.debug(f"LLMClient initialized with model: {self.model}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        json_response: bool = False,
    ) -> Dict[str, Any]:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response (None = no limit)
            temperature: Sampling temperature
            json_response: Whether to expect JSON response

        Returns:
            Response dict with 'content' key or parsed JSON
        """

        def _call_api():
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            # Only add max_tokens if specified
            if max_tokens is not None:
                token_param = self.get_token_param_name(self.model)
                adjusted_tokens = self.adjust_tokens_for_reasoning(
                    self.model, max_tokens
                )
                request_kwargs[token_param] = adjusted_tokens

            if (
                self.is_openai_model(self.model)
                and not self.is_new_openai_model(self.model)
                and json_response
            ):
                request_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content

            if json_response:
                return self.extract_json(content)
            return {"content": content}

        return self._retry_with_backoff(_call_api)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        json_response: bool = False,
    ) -> Dict[str, Any]:
        """
        Simple completion with optional system prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            json_response: Whether to expect JSON response

        Returns:
            Response dict
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_response=json_response,
        )

    def extract_facts(
        self,
        text: str,
        extraction_prompt: str,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Extract structured facts from text.

        Args:
            text: Source text to extract from
            extraction_prompt: Prompt describing what to extract

        Returns:
            Extracted facts as dict
        """
        full_prompt = f"{extraction_prompt}\n\nText to analyze:\n{text}"
        return self.complete(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            json_response=True,
        )


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
