import os
import json
import base64
from openai import OpenAI
from loguru import logger
from LIMP_Poker_V2.config import config
from LIMP_Poker_V2.core.schema import GameState, PhaseType, PlayerState


class BoardParsingAgent:
    def __init__(self):
        self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        self.model = config.LLM_MODEL_NAME

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def parse_state(self, image_path: str, timestamp: float = 0.0) -> GameState:
        """
        Use VLM to extract structured game state from a frame.
        """
        base64_image = self._encode_image(image_path)

        # Load Prompt from file
        if os.path.exists(config.PROMPT_BOARD_PARSING):
            with open(config.PROMPT_BOARD_PARSING, "r") as f:
                prompt = f.read()
        else:
            logger.warning(
                f"Prompt file not found: {config.PROMPT_BOARD_PARSING}, using default."
            )
            prompt = """
            Analyze this poker game frame. Extract strict JSON:
            1. "phase": Pre-flop/Flop/Turn/River/Showdown
            2. "pot": Total pot value (number).
            3. "board": List of community cards ["Ah", "Kd"] etc.
            4. "players": List of visible players:
               - "name": Name or ID.
               - "stack": Current stack size (number or null).
               - "is_active": Boolean.
               - "micro_gestures": { "posture": "...", "hands": "...", "gaze": "...", "occlusion": "..." }
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
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
                max_tokens=500,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)

            # Map to Schema
            players = []
            for p in data.get("players", []):
                # Handle stack being None/String
                stack_val = p.get("stack")
                if isinstance(stack_val, str):
                    # Try parse "123k" -> 123000
                    try:
                        if "k" in stack_val.lower():
                            stack_val = float(stack_val.lower().replace("k", "")) * 1000
                        else:
                            stack_val = float(stack_val)
                    except Exception:
                        stack_val = None

                players.append(
                    PlayerState(
                        name=p.get("name", "Unknown"),
                        stack=stack_val,
                        position=p.get("position"),
                        is_active=p.get("is_active", True),
                        micro_gestures=p.get("micro_gestures", {}),
                    )
                )

            return GameState(
                phase=data.get("phase", PhaseType.PRE_FLOP),  # Fallback default
                board=data.get("board", []),
                pot=data.get("pot", 0),
                players=players,
                timestamp=timestamp,
            )

        except Exception as e:
            logger.error(f"Board parsing failed: {e}")
            # Return empty/safe state
            return GameState(
                phase=PhaseType.PRE_FLOP,
                board=[],
                pot=0,
                players=[],
                timestamp=timestamp,
            )
