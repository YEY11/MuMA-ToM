"""
Posture Analysis Agent
Analyzes behavioral cues to inform reasoning
"""

from typing import Any, Dict
from loguru import logger

from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import (
    QAItem,
    AgentOutput,
    QuestionType,
    SocialGoalType,
)
from .base import BaseReasoningAgent


@AgentRegistry.register_reasoning("posture_agent")
class PostureAgent(BaseReasoningAgent):
    """
    Analyzes behavioral cues (posture, hands, gaze, emotion)
    to provide evidence for/against different options.
    """

    def __init__(self):
        super().__init__()
        # Behavioral indicators for different intents
        self.bluff_indicators = {
            "posture": ["Leaning back", "Neutral"],
            "hands": ["Playing with chips", "Touching face"],
            "gaze": ["Looking away", "Looking down"],
            "emotion": ["Tense", "Uncertain"],
        }
        self.value_indicators = {
            "posture": ["Leaning forward"],
            "hands": ["On table", "Folded"],
            "gaze": ["Staring at opponent", "Looking at board"],
            "emotion": ["Confident", "Neutral"],
        }

    def analyze(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
        **kwargs,
    ) -> AgentOutput:
        """
        Analyze behavioral cues for a question.

        Returns scores for each option based on behavioral evidence.
        """
        # Get behavioral data from context
        behavioral_cues = question.context.behavioral_cues or {}
        decision_time = question.context.decision_time

        # Initialize scores
        option_scores = {opt.key: 0.5 for opt in question.options}

        # Analyze behavioral cues
        bluff_score = 0.0
        value_score = 0.0
        evidence = []

        for player, cues in behavioral_cues.items():
            if isinstance(cues, dict):
                # Check bluff indicators
                for dim, indicators in self.bluff_indicators.items():
                    if cues.get(dim) in indicators or cues.get(f"dominant_{dim}") in indicators:
                        bluff_score += 0.15
                        evidence.append(f"{dim}={cues.get(dim, cues.get(f'dominant_{dim}'))} suggests bluff")

                # Check value indicators
                for dim, indicators in self.value_indicators.items():
                    if cues.get(dim) in indicators or cues.get(f"dominant_{dim}") in indicators:
                        value_score += 0.15
                        evidence.append(f"{dim}={cues.get(dim, cues.get(f'dominant_{dim}'))} suggests value")

                # Fidgeting is often a bluff tell
                if cues.get("fidgeting_detected"):
                    bluff_score += 0.2
                    evidence.append("fidgeting detected - suggests nervousness")

                # Posture/emotion change can indicate deception
                if cues.get("posture_changed") or cues.get("emotion_changed"):
                    bluff_score += 0.1
                    evidence.append("behavioral change detected")

        # Decision time analysis
        if decision_time is not None:
            if decision_time > 10:
                # Long think often indicates difficult decision or acting
                bluff_score += 0.1
                evidence.append(f"long think ({decision_time:.1f}s) - possible bluff")
            elif decision_time < 2:
                # Quick action often indicates routine or planned play
                value_score += 0.1
                evidence.append(f"quick action ({decision_time:.1f}s) - possible value")

        # Map to option scores based on question type
        if question.question_type == QuestionType.INTENT:
            # A=Bluff, B=Value, C=Control (typical)
            option_scores["A"] = 0.33 + bluff_score
            option_scores["B"] = 0.33 + value_score
            option_scores["C"] = 0.33 + (1 - bluff_score - value_score) * 0.3

        elif question.question_type == QuestionType.BINARY:
            # A=Yes (bluff), B=No
            total = bluff_score + value_score + 0.01
            option_scores["A"] = bluff_score / total
            option_scores["B"] = value_score / total

        # Normalize
        total = sum(option_scores.values())
        if total > 0:
            option_scores = {k: v / total for k, v in option_scores.items()}

        return AgentOutput(
            agent_name=self.name,
            timestamp=question.timestamp or 0,
            result={
                "option_scores": option_scores,
                "bluff_score": bluff_score,
                "value_score": value_score,
                "evidence": evidence,
            },
            confidence=min(1.0, bluff_score + value_score),
            reasoning_trace="; ".join(evidence) if evidence else "No clear behavioral signals",
        )

