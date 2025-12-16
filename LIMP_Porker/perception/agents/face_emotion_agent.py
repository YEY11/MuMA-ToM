import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any

load_dotenv()

class FaceEmotionAgent:
    def __init__(self, model_name: str = None):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )
        self.model = model_name or os.getenv("LLM_MODEL_NAME", "gpt-4o")
        logger.info(f"[FaceEmotionAgent] Initialized with model: {self.model}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_scene(self, image_path: str) -> Dict[str, Any]:
        """
        分析全景图中的所有玩家表情。
        不再依赖固定裁剪，而是让模型自动检测画面中的玩家。
        :return: {"players": [{"location": "left", "emotion": ...}, ...]}
        """
        if not os.path.exists(image_path):
            return {"players": [], "error": "Image not found"}

        base64_image = self._encode_image(image_path)
        
        prompt = """
        Analyze the facial expressions of the REAL poker players sitting at the table.
        
        CRITICAL INSTRUCTIONS:
        1. **IGNORE Static UI Photos**: There are static profile photos of players in the bottom corners (UI overlay). DO NOT analyze their emotions. They are not real-time.
        2. **Focus on Real People**: Only analyze the actual humans sitting at the table.
        3. **Exclude Non-Players**: Ignore the dealer or host if visible, unless they are interacting significantly.
        4. **Identify Players**: Use the UI names/photos at the bottom to help identify who is who in the main view (e.g., "Left Player (Hellmuth)", "Right Player (Esfandiari)").
        
        Output a JSON object with a "players" list. For each REAL player found:
        1. "name_guess": Infer name from UI if possible (or "Unknown").
        2. "location": "Left", "Right", "Center", etc.
        3. "primary_emotion": e.g., Neutral, Nervous, Happy, Angry, Focused, Unreadable.
        4. "occlusion": List visible occlusions: "Mask", "Sunglasses", "Hat", "Hand", "None".
        5. "micro_gesture": Any subtle signs? (e.g., "staring down", "heavy breathing", "posture shift").
        6. "arousal": Low/Medium/High.
        
        If the real player's face is fully covered (mask + sunglasses) and no emotion is readable, set "primary_emotion" to "Unreadable" but still record the entry.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Scene face analysis failed: {e}")
            return {"players": [], "error": str(e)}

    # Deprecated: analyze_face (single crop)
    # kept for backward compatibility if needed, but analyze_scene is preferred.

if __name__ == "__main__":
    agent = FaceEmotionAgent()
    # print(agent.analyze_face("path/to/face_crop.jpg"))