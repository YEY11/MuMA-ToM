import os
import json
from loguru import logger
from openai import OpenAI
from LIMP_Poker_V2.config import config


class AudioGTAgent:
    """
    Extracts Ground Truth information from audio commentary.
    Uses Whisper for timestamped transcription and LLM for insight extraction.
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.ASR_API_KEY, base_url=config.ASR_BASE_URL)
        self.model = config.ASR_MODEL_NAME

    def transcribe_and_extract(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return {}

        logger.info(f"Transcribing audio for GT extraction: {audio_path}")

        try:
            # 1. Whisper Transcription (Verbose JSON for timestamps)
            with open(audio_path, "rb") as f:
                # Note: This assumes the endpoint supports standard OpenAI audio API
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-large-v3",  # Or configured model
                    file=f,
                    response_format="verbose_json",
                )

            # 2. Extract Insights per Segment
            # We want to map commentary to the timeline.
            # Instead of one big blob, we return the structured segments.
            # The Reasoning Eval logic will be responsible for matching these segments
            # to the specific Action timestamps.

            # We can also run an LLM pass to "summarize" or "tag" each segment with
            # specific ToM labels (e.g. "Bluff revealed", "Hand strength mentioned").

            segments = transcript.segments  # List of objects with start, end, text

            # Clean and structure
            gt_commentary = []
            for seg in segments:
                # Handle both object (pydantic) and dict access for compatibility
                if isinstance(seg, dict):
                    start = seg.get("start")
                    end = seg.get("end")
                    text = seg.get("text")
                else:
                    start = seg.start
                    end = seg.end
                    text = seg.text

                gt_commentary.append(
                    {"start": start, "end": end, "text": text.strip() if text else ""}
                )

            return {
                "commentary_segments": gt_commentary,
                # In a full implementation, we would add "hole_cards" extraction here
                # by asking LLM to read the transcript and find "Hellmuth has Queen Jack".
                "extracted_facts": self._extract_facts_from_text(transcript.text),
            }

        except Exception as e:
            logger.error(f"Audio GT extraction failed: {e}")
            return {}

    def _extract_facts_from_text(self, full_text):
        # Helper to get high-level facts (Winner, Hole Cards) from full text
        prompt = f"""
        Extract ground truth facts from this poker commentary:
        1. Winner of the hand.
        2. Hole cards of each player (if mentioned).
        3. Key bluffs or strategy reveals.
        
        Text: {full_text[:4000]}...
        
        Output JSON.
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(res.choices[0].message.content)
        except Exception:
            return {}
