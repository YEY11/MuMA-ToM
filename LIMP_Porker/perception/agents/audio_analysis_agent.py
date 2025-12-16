import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, Any

load_dotenv()

class AudioAnalysisAgent:
    def __init__(self):
        # 使用专门的 ASR 配置
        self.client = OpenAI(
            api_key=os.getenv("AIHUBMIX_API_KEY") or os.getenv("LLM_API_KEY"),
            base_url=os.getenv("AIHUBMIX_BASE_URL") or os.getenv("LLM_BASE_URL")
        )
        self.asr_model = "whisper-large-v3"
        self.llm_model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
        logger.info(f"[AudioAnalysisAgent] Initialized. ASR: {self.asr_model}, LLM: {self.llm_model}")

    def transcribe(self, audio_path: str) -> str:
        """
        调用 Whisper API 进行转录
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return ""
            
        logger.info(f"Transcribing {audio_path}...")
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.asr_model,
                    file=audio_file
                )
            return transcription.text
        except Exception as e:
            logger.error(f"ASR failed: {e}")
            return ""

    def analyze_commentary(self, text: str, known_players: list = None) -> Dict[str, Any]:
        """
        从解说文本中提取 ToM 线索，并进行实体对齐。
        :param text: 转录文本
        :param known_players: 从 UI 识别到的标准玩家名字列表 (e.g. ["Hellmuth", "Esfandiari"])
        """
        if not text:
            return {}
            
        player_context = ""
        if known_players:
            player_context = f"The players visible in the video UI are named: {', '.join(known_players)}. Please map any nicknames or first names mentioned in the commentary to these standard UI names."

        prompt = f"""
        Analyze the following poker commentary text. Extract insights about the players' mental states, strategies, or hidden information.
        {player_context}
        
        Text: "{text}"
        
        Output a JSON object with these keys:
        - "context": General game situation (e.g., "High stakes", "History of aggressive play").
        - "player_insights": A list of objects, each containing:
            - "player": The standard UI name of the player (if mappable), otherwise the name mentioned.
            - "belief": What the commentator thinks the player believes.
            - "goal": The player's strategic goal.
            - "emotion": Emotional state.
        
        Only output the JSON.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Commentary analysis failed: {e}")
            return {}

if __name__ == "__main__":
    agent = AudioAnalysisAgent()
    # text = agent.transcribe("path/to/audio.wav")
    # print(agent.analyze_commentary(text))