"""
Vision-Language Model (VLM) Client
Unified interface for image understanding with different backends
"""

import base64
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from LIMP_Poker_V3.config import config

from .base import BaseModelClient


class VLMClient(BaseModelClient):
    """
    Unified VLM client supporting multiple backends (OpenAI, Gemini, etc.)
    through OpenAI-compatible API.
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
        self.model = model or config.VLM_MODEL_NAME

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        logger.debug(f"VLMClient initialized with model: {self.model}")

    def encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        json_response: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze an image with the given prompt.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response (None = no limit, let model complete naturally)
            temperature: Sampling temperature
            json_response: Whether to expect JSON response

        Returns:
            Parsed response as dict (if json_response=True) or raw response

        Raises:
            Exception: If API call fails after retries
        """

        def _call_api():
            base64_image = self.encode_image(image_path)

            request_kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": temperature,
            }

            # Only add max_tokens if specified (None = let model complete naturally)
            if max_tokens is not None:
                token_param = self.get_token_param_name(self.model)
                adjusted_tokens = self.adjust_tokens_for_reasoning(
                    self.model, max_tokens
                )
                request_kwargs[token_param] = adjusted_tokens

            # Only add response_format for legacy OpenAI models (not gpt-5/o1/o3)
            if (
                self.is_openai_model(self.model)
                and not self.is_new_openai_model(self.model)
                and json_response
            ):
                request_kwargs["response_format"] = {"type": "json_object"}

            # Debug: log request params (without image data)
            debug_kwargs = {k: v for k, v in request_kwargs.items() if k != "messages"}
            logger.debug(f"VLM request params: {debug_kwargs}")

            response = self.client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content

            # Debug: log response
            if not content:
                logger.warning(f"VLM returned empty content. Full response: {response}")

            if json_response:
                return self.extract_json(content)
            return {"content": content}

        return self._retry_with_backoff(_call_api)

    def analyze_images(
        self,
        image_paths: List[str],
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        json_response: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze multiple images with the given prompt.

        Args:
            image_paths: List of paths to image files
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response (None = no limit)
            temperature: Sampling temperature
            json_response: Whether to expect JSON response

        Returns:
            Parsed response as dict
        """

        def _call_api():
            content = [{"type": "text", "text": prompt}]

            for image_path in image_paths:
                base64_image = self.encode_image(image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )

            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
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
            response_content = response.choices[0].message.content

            if json_response:
                return self.extract_json(response_content)
            return {"content": response_content}

        return self._retry_with_backoff(_call_api)


# Singleton instance for convenience
_vlm_client: Optional[VLMClient] = None


def get_vlm_client() -> VLMClient:
    """Get or create singleton VLM client"""
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = VLMClient()
    return _vlm_client
