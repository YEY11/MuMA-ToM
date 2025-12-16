import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from typing import List, Dict, Any, Tuple

load_dotenv()

class BoardParsingAgent:
    def __init__(self, model_name: str = None):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )
        self.model = model_name or os.getenv("LLM_MODEL_NAME", "gpt-4o")
        logger.info(f"[BoardParsingAgent] Initialized with model: {self.model}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def parse_game_state(self, image_path: str) -> Dict[str, Any]:
        """
        从全图识别完整的游戏状态：公共牌、玩家筹码、SB/BB位置（通过D标）、底池等。
        参考 muma_tom_pipeline/extract_state.py 的 Prompt。
        """
        base64_image = self._encode_image(image_path)
        
        # 复用并优化 extract_state.py 的 Prompt
        prompt = """
        你将查看一张双人德州扑克的比赛截图。请提取以下信息并输出为严格的 JSON 格式：
        
        1. **blinds**: 识别庄家 (Dealer Button "D")。
           - "D" 标志所在的一方为 SB (Small Blind)。
           - 另一方为 BB (Big Blind)。
           - 记录谁是 SB/BB (例如 "P1" 或 "P2")。
        2. **players**: 识别两名玩家 (P1 左边, P2 右边)。
           - "stack": 剩余筹码数量 (画面上的数字，通常单位为 K 或直接数字，请转换为纯数字)。
           - "name": 玩家名字 (如有)。
        3. **board**: 公共牌 (RankSuit, 如 ["Ah", "Kd", "Ts"])。如果没有则为 []。
        4. **pot**: 当前底池金额 (数字)。
        5. **hole_cards**: 玩家底牌 (如有显示，未知用 "??")。
        
        输出格式示例：
        {
            "blinds": {"SB": "P1", "BB": "P2"},
            "players": [
                {"name": "P1", "seat": 1, "stack": 300000},
                {"name": "P2", "seat": 2, "stack": 240000}
            ],
            "board": ["Ah", "7d", "2c"],
            "pot": 44000,
            "hole_cards": {"P1": ["Ks", "Qs"], "P2": ["??", "??"]}
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a poker game state extractor. Output only JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing game state: {e}")
            return {}

    def parse_board(self, image_path: str) -> List[str]:
        # 为了兼容性保留，内部调用 parse_game_state
        state = self.parse_game_state(image_path)
        return state.get("board", [])

    def parse_stacks(self, image_path: str, player_regions: Dict[str, Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        # 移除 Mock，直接从全图状态中提取
        state = self.parse_game_state(image_path)
        stacks = {}
        for p in state.get("players", []):
            name = p.get("name", "Unknown")
            # 尝试映射 P1/P2
            if "P1" in name or p.get("seat") == 1:
                stacks["P1"] = p.get("stack", 0)
            elif "P2" in name or p.get("seat") == 2:
                stacks["P2"] = p.get("stack", 0)
            else:
                stacks[name] = p.get("stack", 0)
        return stacks

if __name__ == "__main__":
    agent = BoardParsingAgent()
    # print(agent.parse_board("path/to/frame.jpg"))