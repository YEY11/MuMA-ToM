"""
LIMP_Poker_V3 Data Schema
Pydantic models for structured data representation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum


# ========== Enums ==========


class PhaseType(str, Enum):
    """Poker game phases"""

    PRE_FLOP = "Pre-flop"
    FLOP = "Flop"
    TURN = "Turn"
    RIVER = "River"
    SHOWDOWN = "Showdown"
    UNKNOWN = "Unknown"


class ActionType(str, Enum):
    """Player action types"""

    CHECK = "check"
    BET = "bet"
    CALL = "call"
    RAISE = "raise"
    FOLD = "fold"
    ALL_IN = "all-in"
    UNKNOWN = "unknown"


class SocialGoalType(str, Enum):
    """ToM Social Goal types"""

    BLUFF = "bluff"  # 诈唬 - 试图让对手弃牌
    VALUE = "value"  # 价值 - 认为领先，获取价值
    CONTROL = "control"  # 控池 - 控制底池大小
    TRAP = "trap"  # 设陷阱 - 慢打强牌
    UNKNOWN = "unknown"


class FacialEmotionType(str, Enum):
    """Facial emotion categories (VLM inferred)"""

    NEUTRAL = "Neutral"
    TENSE = "Tense"
    CONFIDENT = "Confident"
    UNCERTAIN = "Uncertain"


class QuestionLevel(str, Enum):
    """QA question granularity levels"""

    ACTION = "action"  # 操作级
    PHASE = "phase"  # 阶段级
    GAME = "game"  # 局级 (reserved)


class QuestionType(str, Enum):
    """Types of questions with different option counts"""

    INTENT = "intent"  # 意图判断 - 3选项
    BINARY = "binary"  # 是非判断 - 2选项
    STRATEGY = "strategy"  # 策略预测 - 4选项
    SECOND_ORDER = "second_order"  # 二阶心智 - 3选项


# ========== Behavioral Cues ==========


class BehavioralCues(BaseModel):
    """Micro-posture and behavioral signals"""

    posture: Optional[str] = None  # Leaning forward/back/Neutral
    hands: Optional[str] = None  # Playing with chips/Touching face/Hidden/Folded
    gaze: Optional[str] = None  # Staring at opponent/Looking at board/Looking down
    occlusion: Optional[str] = None  # Sunglasses/Hat/Mask/None
    facial_emotion: Optional[FacialEmotionType] = None  # VLM inferred emotion


# ========== Game State ==========


class PlayerState(BaseModel):
    """State of a single player at a point in time"""

    name: str
    position: Optional[str] = None  # SB/BB
    stack: Optional[float] = None
    hole_cards: Optional[List[str]] = None  # Only in GT or Showdown
    is_active: bool = True  # Has not folded
    behavioral_cues: Optional[BehavioralCues] = None


class GameState(BaseModel):
    """Complete game state at a single frame"""

    timestamp: float
    phase: PhaseType = PhaseType.UNKNOWN
    board: List[str] = Field(default_factory=list)  # Community cards
    pot: Optional[float] = None
    players: List[PlayerState] = Field(default_factory=list)


# ========== Action Events ==========


class ActionEvent(BaseModel):
    """A single player action with context"""

    timestamp: float
    player_name: str
    action_type: ActionType
    amount: float = 0.0

    # Decision timing analysis
    decision_start_time: Optional[float] = None  # When previous action ended
    duration: Optional[float] = None  # Decision time in seconds

    # Behavioral analysis over decision interval
    decision_frame_count: Optional[int] = None
    behavioral_sequence: Optional[List[BehavioralCues]] = None
    behavioral_summary: Optional[Dict[str, Any]] = None

    # Visual context snapshot
    visual_context: Optional[Dict[str, Any]] = None

    # Detection metadata
    detection_source: Literal["visual", "audio_gt", "inferred"] = "visual"
    confidence: float = 1.0


# ========== Phase and Episode Data ==========


class PhaseData(BaseModel):
    """Data for a single game phase"""

    phase: PhaseType
    start_time: float
    end_time: float
    actions: List[ActionEvent] = Field(default_factory=list)
    initial_state: Optional[GameState] = None
    final_state: Optional[GameState] = None


class EpisodeData(BaseModel):
    """Complete data for one poker hand/game"""

    episode_id: str
    protocol: Literal["audience", "player"]
    meta: Dict[str, Any] = Field(default_factory=dict)
    timeline: List[PhaseData] = Field(default_factory=list)
    ground_truth: Dict[str, Any] = Field(default_factory=dict)


# ========== QA Dataset Schema ==========


class QAOption(BaseModel):
    """A single option in a question"""

    key: str  # A, B, C, D
    text: str
    is_correct: bool = False


class QAContext(BaseModel):
    """Context information for a question"""

    phase: Optional[PhaseType] = None
    board: List[str] = Field(default_factory=list)
    pot: Optional[float] = None
    action: Optional[Dict[str, Any]] = None
    action_sequence: Optional[List[Dict[str, Any]]] = None
    visible_cards: Optional[Dict[str, List[str]]] = None  # For audience protocol
    behavioral_cues: Optional[Dict[str, BehavioralCues]] = None
    decision_time: Optional[float] = None


class ToMLabels(BaseModel):
    """Theory of Mind labels for a question"""

    social_goal: Optional[SocialGoalType] = None
    belief: Optional[str] = None  # Natural language description of belief
    believed_goal: Optional[str] = None  # Reserved for future


class QAItem(BaseModel):
    """A single QA item in the dataset"""

    id: str
    level: QuestionLevel
    question_type: QuestionType
    protocol: Literal["audience", "player"]

    timestamp: Optional[float] = None
    phase: Optional[PhaseType] = None

    context: QAContext
    question: str
    options: List[QAOption]
    answer: str  # The correct option key (A/B/C/D)
    answer_source: str  # "audio_commentary" / "expert_annotation" / "rule_based"

    # ToM labels
    tom_labels: Optional[ToMLabels] = None

    # Metadata
    difficulty: Optional[str] = None  # easy/medium/hard


class QADataset(BaseModel):
    """Complete QA dataset for an episode"""

    episode_id: str
    protocol: Literal["audience", "player"]
    version: str = "1.0"
    questions: List[QAItem] = Field(default_factory=list)

    def get_by_level(self, level: QuestionLevel) -> List[QAItem]:
        """Filter questions by level"""
        return [q for q in self.questions if q.level == level]

    def get_by_type(self, question_type: QuestionType) -> List[QAItem]:
        """Filter questions by type"""
        return [q for q in self.questions if q.question_type == question_type]


# ========== Reasoning Output ==========


class AgentOutput(BaseModel):
    """Output from a single reasoning agent"""

    agent_name: str
    timestamp: float
    result: Dict[str, Any]
    confidence: float = 1.0
    reasoning_trace: Optional[str] = None  # For explainability


class ReasoningResult(BaseModel):
    """Complete reasoning result with all agent outputs"""

    question_id: str
    predicted_answer: str
    confidence: float
    agent_outputs: List[AgentOutput] = Field(default_factory=list)
    aggregation_method: str = "weighted_sum"
    final_scores: Dict[str, float] = Field(default_factory=dict)  # Option -> score

