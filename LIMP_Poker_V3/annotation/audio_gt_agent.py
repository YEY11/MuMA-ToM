"""
Audio Ground Truth Agent
Extracts ground truth information from poker commentary audio
"""

import os
from typing import Any, Dict, List, Optional

from loguru import logger

from LIMP_Poker_V3.models import ASRClient, LLMClient


class AudioGTAgent:
    """
    Extracts Ground Truth from audio commentary.
    Uses Whisper for transcription and LLM for structured extraction.

    Note: Audio is ONLY used for GT generation, NOT for inference.
    This avoids information leakage during reasoning.
    """

    def __init__(self, use_local_asr: bool = False):
        self.asr = ASRClient(use_local=use_local_asr)
        self.llm = LLMClient()

    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        Process audio file to extract ground truth.

        Args:
            audio_path: Path to audio.wav file

        Returns:
            Dict containing:
            - transcript: Full text transcript
            - segments: Timestamped segments
            - facts: Extracted facts (players, hole cards, actions, winner)
            - commentary_insights: Strategic insights from commentators
        """
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return {"error": "Audio file not found"}

        logger.info(f"Processing audio for GT extraction: {audio_path}")

        try:
            # 1. Transcribe with timestamps
            transcript_data = self.asr.transcribe(audio_path, language="en")

            # 2. Extract structured facts
            facts = self._extract_facts(transcript_data.get("text", ""))

            # 3. Extract action-level GT from segments
            action_gt = self._extract_action_gt(transcript_data.get("segments", []))

            return {
                "transcript": transcript_data.get("text", ""),
                "segments": transcript_data.get("segments", []),
                "facts": facts,
                "action_gt": action_gt,
            }

        except Exception as e:
            logger.error(f"Audio GT extraction failed: {e}")
            return {"error": str(e)}

    def _extract_facts(self, text: str) -> Dict[str, Any]:
        """
        Extract structured facts from transcript using LLM.
        """
        if not text:
            return {}

        prompt = f"""Analyze this poker commentary transcript and extract ground truth facts.

Transcript:
{text[:6000]}

Extract and output as JSON:
{{
    "players": [
        {{"name": "...", "hole_cards": ["Ah", "Kd"], "position": "SB/BB"}}
    ],
    "winner": "player name or null if not mentioned",
    "final_hand": "winning hand description if mentioned",
    "key_moments": [
        {{"timestamp_approx": "description", "event": "bluff/value/fold/etc", "player": "name"}}
    ],
    "bluff_mentions": [
        {{"player": "name", "description": "what was said about their bluff"}}
    ],
    "strategy_insights": ["any strategic comments from commentators"]
}}
"""

        try:
            return self.llm.complete(
                prompt=prompt,
                temperature=0.0,
                json_response=True,
            )
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return {}

    def _extract_action_gt(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract action-level ground truth from timestamped segments.
        Maps commentary to specific time ranges for action labeling.
        """
        action_gt = []

        for seg in segments:
            text = seg.get("text", "").lower()
            start = seg.get("start", 0)
            end = seg.get("end", 0)

            gt_entry = {
                "start": start,
                "end": end,
                "text": seg.get("text", ""),
                "labels": {},
            }

            # Simple keyword-based detection
            if any(word in text for word in ["bluff", "bluffing", "representing"]):
                gt_entry["labels"]["is_bluff"] = True
            if any(word in text for word in ["value", "has it", "holding"]):
                gt_entry["labels"]["is_value"] = True
            if any(word in text for word in ["fold", "folding", "gives up"]):
                gt_entry["labels"]["action"] = "fold"
            if any(word in text for word in ["all in", "all-in", "shoves"]):
                gt_entry["labels"]["action"] = "all-in"
            if any(word in text for word in ["raises", "raise"]):
                gt_entry["labels"]["action"] = "raise"
            if any(word in text for word in ["calls", "call"]):
                gt_entry["labels"]["action"] = "call"
            if any(word in text for word in ["checks", "check"]):
                gt_entry["labels"]["action"] = "check"

            if gt_entry["labels"]:
                action_gt.append(gt_entry)

        return action_gt

    def get_gt_for_timestamp(
        self,
        gt_data: Dict[str, Any],
        timestamp: float,
        window: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Find ground truth relevant to a specific timestamp.

        Args:
            gt_data: Ground truth data from process()
            timestamp: Target timestamp in seconds
            window: Time window to search (before and after)

        Returns:
            Relevant GT data or None
        """
        action_gt = gt_data.get("action_gt", [])

        for entry in action_gt:
            start = entry.get("start", 0)
            end = entry.get("end", 0)

            if start - window <= timestamp <= end + window:
                return entry

        return None
