import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from loguru import logger

load_dotenv()

class PhaseSegmentationAgent:
    def __init__(self, model_name: str = None):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )
        self.model = model_name or os.getenv("LLM_MODEL_NAME", "gpt-4o")
        logger.info(f"[PhaseSegmentationAgent] Initialized with model: {self.model}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def detect_phases(self, frames_dir: str, sample_interval: int = 5) -> Dict[str, Dict[str, int]]:
        """
        通过采样帧检测公共牌数量变化，从而确定游戏阶段。
        :param frames_dir: 帧图像目录
        :param sample_interval: 采样间隔（帧数），默认每5帧看一次以节省Token
        :return: 阶段字典，例如 {"Pre-flop": {"start": 0, "end": 20}, ...}
        """
        frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        if not frames:
            logger.error("No frames found!")
            return {}

        logger.info(f"Scanning {len(frames)} frames with interval {sample_interval}...")
        
        # 简化策略：只对采样帧进行识别，记录公共牌数量
        # 0 cards -> Pre-flop
        # 3 cards -> Flop
        # 4 cards -> Turn
        # 5 cards -> River
        
        timeline = []
        
        for i in range(0, len(frames), sample_interval):
            frame_path = frames[i]
            card_count = self._count_board_cards(frame_path)
            timeline.append((i, card_count))
            logger.debug(f"Frame {i}: {card_count} cards")

        # 后处理：平滑并生成区间
        phases = self._timeline_to_phases(timeline, len(frames))
        return phases

    def _count_board_cards(self, image_path: str) -> int:
        base64_image = self._encode_image(image_path)
        
        prompt = """
        Look at the center of the poker table. How many community cards (board cards) are face up?
        Only output a single integer (0, 3, 4, or 5).
        If unsure or between phases (animation), output the previous number you'd expect.
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
                max_tokens=10,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            # 简单的清洗，防止模型输出 extra text
            digits = [c for c in content if c.isdigit()]
            if digits:
                return int(digits[0])
            return 0
        except Exception as e:
            logger.warning(f"Failed to count cards for {image_path}: {e}")
            return 0

    def _timeline_to_phases(self, timeline: List[Tuple[int, int]], total_frames: int) -> Dict[str, Dict[str, int]]:
        # 简单的状态机
        # 默认顺序: Pre-flop (0) -> Flop (3) -> Turn (4) -> River (5)
        # 注意：有时候可能会跳过（如直接结束），或者识别错误。这里做简单的平滑。
        
        phases = {}
        current_phase = "Pre-flop"
        start_frame = 0
        
        # 映射关系
        count_to_phase = {0: "Pre-flop", 3: "Flop", 4: "Turn", 5: "River"}
        
        # 为了去噪，我们可以要求连续 N 次检测到新数量才切换，这里简化处理：
        # 只要检测到 > 当前数量，就认为是新阶段开始
        
        last_count = 0
        phases[current_phase] = {"start": 0, "end": total_frames - 1} # 初始化

        for frame_idx, count in timeline:
            if count > last_count:
                # 状态切换
                if count in count_to_phase:
                    new_phase = count_to_phase[count]
                    # 结束上一个阶段
                    phases[current_phase]["end"] = frame_idx - 1
                    # 开启新阶段
                    current_phase = new_phase
                    phases[current_phase] = {"start": frame_idx, "end": total_frames - 1}
                    last_count = count
                    logger.info(f"Phase change detected: {current_phase} at frame {frame_idx}")

        return phases

if __name__ == "__main__":
    # Test stub
    agent = PhaseSegmentationAgent()
    # print(agent.detect_phases("/path/to/frames"))