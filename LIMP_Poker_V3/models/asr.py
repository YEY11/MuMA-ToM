"""
Automatic Speech Recognition (ASR) Client
Unified interface for audio transcription with Whisper
"""

import os
from typing import Any, Dict, Optional

from loguru import logger

from LIMP_Poker_V3.config import config

from .base import BaseModelClient


class ASRClient(BaseModelClient):
    """
    ASR client using OpenAI Whisper or compatible APIs.
    Falls back to local whisper if API not available.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        use_local: bool = False,
        max_retries: int = 3,
    ):
        super().__init__(max_retries=max_retries)

        self.api_key = (
            api_key or config.ASR_API_KEY or config.API_KEY or config.LLM_API_KEY
        )
        self.base_url = (
            base_url
            or config.ASR_BASE_URL
            or config.API_BASE_URL
            or config.LLM_BASE_URL
        )
        self.model = model or config.ASR_MODEL_NAME
        self.use_local = use_local

        self._local_model = None

        if not use_local:
            try:
                from openai import OpenAI

                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                logger.debug(f"ASRClient initialized with API model: {self.model}")
            except Exception as e:
                logger.warning(
                    f"Failed to init OpenAI client: {e}, falling back to local"
                )
                self.use_local = True

        if self.use_local:
            logger.debug("ASRClient using local whisper model")

    def _get_local_model(self):
        """Lazy load local whisper model"""
        if self._local_model is None:
            try:
                import whisper

                self._local_model = whisper.load_model("base")
                logger.info("Loaded local whisper model: base")
            except ImportError:
                raise ImportError(
                    "Local whisper not available. Install with: pip install openai-whisper"
                )
        return self._local_model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        response_format: str = "verbose_json",
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            language: Language code (e.g., 'en', 'zh')
            response_format: Response format (json, text, verbose_json)

        Returns:
            Dict with 'text' and optionally 'segments' keys
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.use_local:
            return self._transcribe_local(audio_path, language)
        else:
            return self._transcribe_api(audio_path, language, response_format)

    def _transcribe_api(
        self,
        audio_path: str,
        language: Optional[str],
        response_format: str,
    ) -> Dict[str, Any]:
        """Transcribe using OpenAI API"""

        def _call_api():
            with open(audio_path, "rb") as audio_file:
                kwargs = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": response_format,
                }
                if language:
                    kwargs["language"] = language

                response = self.client.audio.transcriptions.create(**kwargs)

            # Handle different response formats
            if hasattr(response, "text"):
                result = {"text": response.text}
                if hasattr(response, "segments"):
                    result["segments"] = [
                        {
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                        }
                        for s in response.segments
                    ]
                return result
            elif isinstance(response, str):
                return {"text": response}
            else:
                return {"text": str(response)}

        return self._retry_with_backoff(_call_api)

    def _transcribe_local(
        self,
        audio_path: str,
        language: Optional[str],
    ) -> Dict[str, Any]:
        """Transcribe using local whisper model"""
        model = self._get_local_model()

        kwargs = {"fp16": False}
        if language:
            kwargs["language"] = language

        result = model.transcribe(audio_path, **kwargs)

        return {
            "text": result["text"],
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
                for s in result.get("segments", [])
            ],
            "language": result.get("language"),
        }


# Singleton instance
_asr_client: Optional[ASRClient] = None


def get_asr_client(use_local: bool = False) -> ASRClient:
    """Get or create singleton ASR client"""
    global _asr_client
    if _asr_client is None:
        _asr_client = ASRClient(use_local=use_local)
    return _asr_client
