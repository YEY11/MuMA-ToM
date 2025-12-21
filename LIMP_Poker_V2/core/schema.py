from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum


class PhaseType(str, Enum):
    PRE_FLOP = "Pre-flop"
    FLOP = "Flop"
    TURN = "Turn"
    RIVER = "River"
    SHOWDOWN = "Showdown"


class ActionType(str, Enum):
    CHECK = "check"
    BET = "bet"
    CALL = "call"
    RAISE = "raise"
    FOLD = "fold"
    ALL_IN = "all-in"
    UNKNOWN = "unknown"


class PlayerState(BaseModel):
    name: str
    position: Optional[str] = None
    stack: Optional[float] = None  # Stack might be unreadable
    hole_cards: Optional[List[str]] = None  # Only available in GT or Showdown
    is_active: bool = True
    micro_gestures: Optional[Dict[str, Any]] = (
        None  # New: Posture, Hands, Gaze, Occlusion
    )


class GameState(BaseModel):
    phase: PhaseType
    board: List[str]
    pot: Optional[float] = None
    players: List[PlayerState]
    timestamp: float


class ActionEvent(BaseModel):
    timestamp: float
    player_name: str
    action_type: ActionType
    amount: float = 0.0
    duration: Optional[float] = None  # Decision time
    visual_context: Optional[Dict[str, Any]] = None  # Snapshot of board/pot
    emotion_context: Optional[Dict[str, Any]] = None  # Emotion analysis


class PhaseData(BaseModel):
    phase: PhaseType
    start_time: float
    end_time: float
    actions: List[ActionEvent] = []
    initial_state: GameState
    final_state: GameState


class EpisodeData(BaseModel):
    episode_id: str
    meta: Dict[str, Any]
    timeline: List[PhaseData]
    ground_truth: Dict[str, Any]  # Winner, full hands, commentary insights
