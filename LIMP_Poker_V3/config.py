"""
LIMP_Poker_V3 Configuration Management
Supports pluggable agent architecture and ablation experiments
"""

import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


class Config:
    """Central configuration for LIMP_Poker_V3 pipeline"""

    # ========== LLM Settings ==========
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o")

    # ========== ASR Settings (for Ground Truth) ==========
    ASR_API_KEY: str = os.getenv("ASR_API_KEY", os.getenv("LLM_API_KEY", ""))
    ASR_BASE_URL: str = os.getenv("ASR_BASE_URL", os.getenv("LLM_BASE_URL", ""))
    ASR_MODEL_NAME: str = os.getenv("ASR_MODEL_NAME", "whisper-large-v3")

    # ========== Protocol Mode ==========
    # "audience": Both players' hole cards visible (complete information game)
    # "player": Opponent's hole cards hidden (incomplete information game)
    PROTOCOL_MODE: str = os.getenv("PROTOCOL_MODE", "audience")

    # ========== Sampling Settings ==========
    FPS: int = int(os.getenv("FPS", "1"))
    SAMPLING_INTERVAL: int = int(os.getenv("SAMPLING_INTERVAL", "1"))

    # ========== Agent Configuration (Pluggable for Ablation) ==========
    AGENT_CONFIG: Dict[str, bool] = {
        # Perception Layer (usually always enabled)
        "board_agent": os.getenv("AGENT_BOARD", "True").lower() == "true",
        "action_detector": os.getenv("AGENT_ACTION", "True").lower() == "true",
        # Reasoning Layer (toggle for ablation experiments)
        "posture_agent": os.getenv("AGENT_POSTURE", "True").lower() == "true",
        "equity_agent": os.getenv("AGENT_EQUITY", "False").lower() == "true",
        "tom_belief_agent": os.getenv("AGENT_TOM_BELIEF", "True").lower() == "true",
        "tom_social_agent": os.getenv("AGENT_TOM_SOCIAL", "True").lower() == "true",
    }

    # ========== Feature Flags ==========
    USE_DECISION_TIME: bool = os.getenv("USE_DECISION_TIME", "True").lower() == "true"
    USE_POSTURE_SEQUENCE: bool = (
        os.getenv("USE_POSTURE_SEQUENCE", "True").lower() == "true"
    )
    USE_FACIAL_EMOTION: bool = (
        os.getenv("USE_FACIAL_EMOTION", "True").lower() == "true"
    )

    # ========== QA Generation Settings ==========
    QA_LEVELS: list = os.getenv("QA_LEVELS", "action,phase").split(",")

    # ========== Paths ==========
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    PROMPTS_DIR: str = os.path.join(BASE_DIR, "prompts")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")

    # Ensure directories exist
    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.PROMPTS_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    # ========== Helper Methods ==========
    @classmethod
    def is_agent_enabled(cls, agent_name: str) -> bool:
        """Check if a specific agent is enabled"""
        return cls.AGENT_CONFIG.get(agent_name, False)

    @classmethod
    def get_enabled_agents(cls, layer: str = "all") -> list:
        """Get list of enabled agent names"""
        perception_agents = ["board_agent", "action_detector"]
        reasoning_agents = [
            "posture_agent",
            "equity_agent",
            "tom_belief_agent",
            "tom_social_agent",
        ]

        if layer == "perception":
            agents = perception_agents
        elif layer == "reasoning":
            agents = reasoning_agents
        else:
            agents = perception_agents + reasoning_agents

        return [name for name in agents if cls.AGENT_CONFIG.get(name, False)]

    @classmethod
    def print_config(cls):
        """Print current configuration for debugging"""
        print("=" * 50)
        print("LIMP_Poker_V3 Configuration")
        print("=" * 50)
        print(f"Protocol Mode: {cls.PROTOCOL_MODE}")
        print(f"FPS: {cls.FPS}")
        print(f"QA Levels: {cls.QA_LEVELS}")
        print("\nEnabled Agents:")
        for name, enabled in cls.AGENT_CONFIG.items():
            status = "✓" if enabled else "✗"
            print(f"  [{status}] {name}")
        print("\nFeature Flags:")
        print(f"  Decision Time: {cls.USE_DECISION_TIME}")
        print(f"  Posture Sequence: {cls.USE_POSTURE_SEQUENCE}")
        print(f"  Facial Emotion: {cls.USE_FACIAL_EMOTION}")
        print("=" * 50)


# Create singleton instance
config = Config()
config.ensure_dirs()

