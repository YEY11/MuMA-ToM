import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


class Config:
    # LLM Settings
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")

    # ASR Settings
    ASR_API_KEY = os.getenv("ASR_API_KEY")
    ASR_BASE_URL = os.getenv("ASR_BASE_URL")
    ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "whisper-1")

    # Poker Settings
    POKER_DATASET_ROOT = os.getenv("POKER_DATASET_ROOT")
    POKER_EPISODE = os.getenv("POKER_EPISODE")

    # Perception Settings
    SAMPLING_INTERVAL = int(os.getenv("SAMPLING_INTERVAL", 1))  # Seconds
    FPS = int(os.getenv("FPS", 1))  # Frames extraction fps
    USE_AUDIO_FOR_INFERENCE = (
        os.getenv("USE_AUDIO_FOR_INFERENCE", "False").lower() == "true"
    )
    USE_EMOTION_FOR_INFERENCE = (
        os.getenv("USE_EMOTION_FOR_INFERENCE", "True").lower() == "true"
    )

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    # Prompt Files
    PROMPT_BOARD_PARSING = os.path.join(PROMPTS_DIR, "board_parsing.txt")

    # Ensure directories exist
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


config = Config()
