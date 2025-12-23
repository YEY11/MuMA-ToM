"""
Base model client with common utilities
"""

import json
import re
import time
from typing import Any, Dict

from loguru import logger


class BaseModelClient:
    """Base class for model clients with common utilities"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed: {e}")

        raise last_exception

    @staticmethod
    def extract_json(content: str) -> Dict[str, Any]:
        """
        Extract JSON from response content.
        Handles both raw JSON and markdown code blocks.

        Args:
            content: Response content string

        Returns:
            Parsed JSON as dict

        Raises:
            ValueError: If JSON extraction fails
        """
        if not content:
            raise ValueError("Empty response content")

        content = content.strip()

        # Try to parse as raw JSON first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        # Try to find JSON object in the content
        brace_match = re.search(r"\{[\s\S]*\}", content)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response: {content[:200]}...")

    @staticmethod
    def is_openai_model(model_name: str) -> bool:
        """Check if model is an OpenAI model"""
        return model_name.startswith("gpt-") or model_name.startswith("o1")

    @staticmethod
    def is_new_openai_model(model_name: str) -> bool:
        """Check if model uses new OpenAI API parameters (gpt-5, o1, o3, etc.)"""
        # These models use max_completion_tokens instead of max_tokens
        new_models = ["gpt-5", "o1", "o3"]
        return any(model_name.startswith(prefix) for prefix in new_models)

    @staticmethod
    def is_reasoning_model(model_name: str) -> bool:
        """Check if model is a reasoning model that uses internal reasoning tokens"""
        # These models spend tokens on internal reasoning before output
        reasoning_models = ["gpt-5", "o1", "o3"]
        return any(model_name.startswith(prefix) for prefix in reasoning_models)

    @staticmethod
    def adjust_tokens_for_reasoning(model_name: str, requested_tokens: int) -> int:
        """
        Adjust token count for reasoning models.
        Reasoning models need extra tokens for internal reasoning.
        """
        if BaseModelClient.is_reasoning_model(model_name):
            # Reasoning models typically use 50-80% of tokens for reasoning
            # Multiply by 3 to ensure enough tokens for output
            return max(requested_tokens * 3, 2000)
        return requested_tokens

    @staticmethod
    def is_gemini_model(model_name: str) -> bool:
        """Check if model is a Gemini model"""
        return "gemini" in model_name.lower()

    @staticmethod
    def is_claude_model(model_name: str) -> bool:
        """Check if model is a Claude model"""
        return "claude" in model_name.lower()

    @staticmethod
    def get_token_param_name(model_name: str) -> str:
        """Get the correct token parameter name for the model"""
        if BaseModelClient.is_new_openai_model(model_name):
            return "max_completion_tokens"
        return "max_tokens"
