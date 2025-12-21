"""
Board Parsing Agent
Uses VLM to extract structured game state from poker frames
"""

import os
import json
import base64
from typing import Dict, Any, Optional
from openai import OpenAI
from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import (
    GameState,
    PlayerState,
    BehavioralCues,
    PhaseType,
    FacialEmotionType,
)
from .base import BasePerceptionAgent


# Default prompt for board parsing
DEFAULT_PROMPT = """Analyze this poker game frame. Extract strict JSON:

1. "phase": Pre-flop/Flop/Turn/River/Showdown/Unknown
2. "pot": Total pot value (number). If unsure, output null.
3. "board": List of community cards ["Ah", "Kd", "7c"] etc. Empty list if pre-flop.
4. "players": List of visible players. For EACH player:
   - "name": Name or identifier visible on screen
   - "stack": Current stack size (number). If unreadable, output null.
   - "position": "SB" or "BB" (look for Dealer button)
   - "is_active": Boolean (has not folded, still in hand)
   - "behavioral_cues":
     - "posture": "Leaning forward" / "Leaning back" / "Neutral"
     - "hands": "Playing with chips" / "Touching face" / "Hidden" / "Folded" / "On table"
     - "gaze": "Staring at opponent" / "Looking at board" / "Looking down" / "Looking away"
     - "occlusion": "Sunglasses" / "Hat" / "Mask" / "None"
     - "facial_emotion": "Neutral" / "Tense" / "Confident" / "Uncertain"

Output ONLY the JSON object, no markdown formatting."""


@AgentRegistry.register_perception("board_agent")
class BoardAgent(BasePerceptionAgent):
    """
    VLM-based agent for extracting game state from poker frames.
    Extracts: phase, board, pot, players (with behavioral cues).
    """

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model = config.LLM_MODEL_NAME
        self._load_prompt()

    def _load_prompt(self):
        """Load prompt from file or use default"""
        prompt_path = os.path.join(config.PROMPTS_DIR, "board_parsing.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                self.prompt = f.read()
        else:
            self.prompt = DEFAULT_PROMPT
            # Save default prompt for future customization
            os.makedirs(config.PROMPTS_DIR, exist_ok=True)
            with open(prompt_path, "w") as f:
                f.write(DEFAULT_PROMPT)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process(
        self, image_path: str, timestamp: float, **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single frame and extract game state.

        Args:
            image_path: Path to the frame image
            timestamp: Timestamp of the frame

        Returns:
            Dict containing raw extracted data
        """
        try:
            base64_image = self._encode_image(image_path)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=800,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            data = json.loads(response.choices[0].message.content)
            data["timestamp"] = timestamp
            data["_source"] = "vlm"
            return data

        except Exception as e:
            logger.error(f"Board parsing failed at {timestamp}s: {e}")
            return {
                "phase": "Unknown",
                "board": [],
                "pot": None,
                "players": [],
                "timestamp": timestamp,
                "_source": "error",
                "_error": str(e),
            }

    def parse_to_game_state(
        self, raw_data: Dict[str, Any], timestamp: float
    ) -> GameState:
        """
        Convert raw VLM output to structured GameState.

        Args:
            raw_data: Raw dict from VLM
            timestamp: Frame timestamp

        Returns:
            GameState object
        """
        players = []
        for p in raw_data.get("players", []):
            # Parse stack (handle string formats like "123k")
            stack_val = p.get("stack")
            if isinstance(stack_val, str):
                try:
                    if "k" in stack_val.lower():
                        stack_val = float(stack_val.lower().replace("k", "")) * 1000
                    elif "m" in stack_val.lower():
                        stack_val = float(stack_val.lower().replace("m", "")) * 1000000
                    else:
                        stack_val = float(stack_val.replace(",", ""))
                except Exception:
                    stack_val = None

            # Parse behavioral cues
            cues_data = p.get("behavioral_cues", {})
            facial_emotion = cues_data.get("facial_emotion")
            if facial_emotion and config.USE_FACIAL_EMOTION:
                try:
                    facial_emotion = FacialEmotionType(facial_emotion)
                except ValueError:
                    facial_emotion = None
            else:
                facial_emotion = None

            behavioral_cues = BehavioralCues(
                posture=cues_data.get("posture"),
                hands=cues_data.get("hands"),
                gaze=cues_data.get("gaze"),
                occlusion=cues_data.get("occlusion"),
                facial_emotion=facial_emotion,
            )

            players.append(
                PlayerState(
                    name=p.get("name", "Unknown"),
                    stack=stack_val,
                    position=p.get("position"),
                    is_active=p.get("is_active", True),
                    behavioral_cues=behavioral_cues,
                )
            )

        # Parse phase
        phase_str = raw_data.get("phase", "Unknown")
        try:
            phase = PhaseType(phase_str)
        except ValueError:
            phase = PhaseType.UNKNOWN

        return GameState(
            timestamp=timestamp,
            phase=phase,
            board=raw_data.get("board", []),
            pot=raw_data.get("pot"),
            players=players,
        )

