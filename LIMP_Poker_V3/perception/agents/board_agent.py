"""
Board Parsing Agent
Uses VLM to extract structured game state from poker frames
"""

import os
from typing import Any, Dict

from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import (
    BehavioralCues,
    FacialEmotionType,
    GameState,
    PhaseType,
    PlayerState,
)
from LIMP_Poker_V3.models import VLMClient

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
        self.vlm = VLMClient()
        self._load_prompt()

    def _load_prompt(self):
        """Load prompt from file or use default"""
        prompt_path = os.path.join(config.PROMPTS_DIR, "board_parsing.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                self.prompt = f.read()
        else:
            self.prompt = DEFAULT_PROMPT
            os.makedirs(config.PROMPTS_DIR, exist_ok=True)
            with open(prompt_path, "w") as f:
                f.write(DEFAULT_PROMPT)

    def process(self, image_path: str, timestamp: float, **kwargs) -> Dict[str, Any]:
        """
        Process a single frame and extract game state.

        Args:
            image_path: Path to the frame image
            timestamp: Timestamp of the frame

        Returns:
            Dict containing raw extracted data
        """
        try:
            # Don't set max_tokens - let model complete JSON naturally to avoid truncation
            data = self.vlm.analyze_image(
                image_path=image_path,
                prompt=self.prompt,
                temperature=0.0,
                json_response=True,
            )
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
        players_data = raw_data.get("players") or []
        for p in players_data:
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
            cues_data = p.get("behavioral_cues") or {}
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

            # Handle null values with explicit fallbacks
            player_name = p.get("name") or "Unknown"
            is_active = p.get("is_active")
            if is_active is None:
                is_active = True

            players.append(
                PlayerState(
                    name=player_name,
                    stack=stack_val,
                    position=p.get("position"),
                    is_active=is_active,
                    behavioral_cues=behavioral_cues,
                )
            )

        # Parse phase
        frame_type = raw_data.get("frame_type", "standard")
        if frame_type == "transition":
            phase = PhaseType.UNKNOWN
            logger.debug(f"Transition frame detected at {timestamp}s")
        else:
            phase_str = raw_data.get("phase", "Unknown")
            try:
                phase = PhaseType(phase_str)
            except ValueError:
                phase = PhaseType.UNKNOWN

        return GameState(
            timestamp=timestamp,
            phase=phase,
            board=raw_data.get("board", []) or [],
            pot=raw_data.get("pot"),
            players=players,
        )
