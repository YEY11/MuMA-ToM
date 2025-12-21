"""
Action-Level Question Templates
Templates for generating questions about individual actions
"""

from typing import Dict, List, Any
from LIMP_Poker_V3.core.schema import (
    ActionEvent,
    ActionType,
    QuestionType,
    SocialGoalType,
    QAOption,
)


class ActionLevelTemplates:
    """
    Templates for action-level questions.
    Each action can generate multiple question types.
    """

    @staticmethod
    def intent_question(
        player_name: str,
        action_type: ActionType,
        amount: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate intent judgment question (3 options).
        "Is this player bluffing, value betting, or controlling?"
        """
        action_desc = f"{action_type.value}"
        if amount > 0:
            action_desc += f" ${amount:,.0f}"

        question = f"{player_name} 的这次 {action_desc} 最可能的意图是什么？"

        options = [
            QAOption(
                key="A",
                text=f"Bluff（诈唬）- {player_name} 试图通过激进下注逼对手弃牌，手牌可能较弱",
                is_correct=False,
            ),
            QAOption(
                key="B",
                text=f"Value（价值）- {player_name} 认为自己领先，希望对手跟注以获取更多价值",
                is_correct=False,
            ),
            QAOption(
                key="C",
                text=f"Control（控池）- {player_name} 试图控制底池大小，保持灵活性",
                is_correct=False,
            ),
        ]

        return {
            "question": question,
            "question_type": QuestionType.INTENT,
            "options": options,
            "option_count": 3,
        }

    @staticmethod
    def binary_bluff_question(
        player_name: str,
        action_type: ActionType,
        amount: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate binary bluff question (2 options).
        "Is this player bluffing or not?"
        """
        action_desc = f"{action_type.value}"
        if amount > 0:
            action_desc += f" ${amount:,.0f}"

        question = f"{player_name} 的这次 {action_desc} 是否是诈唬（Bluff）？"

        options = [
            QAOption(
                key="A",
                text=f"是 - {player_name} 正在诈唬，手牌实力可能不如表现出的那么强",
                is_correct=False,
            ),
            QAOption(
                key="B",
                text=f"否 - {player_name} 不是在诈唬，这是基于手牌实力的正常下注",
                is_correct=False,
            ),
        ]

        return {
            "question": question,
            "question_type": QuestionType.BINARY,
            "options": options,
            "option_count": 2,
        }

    @staticmethod
    def strategy_prediction_question(
        player_name: str,
        phase: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate strategy prediction question (4 options).
        "What will this player do next?"
        """
        question = f"在当前局面下，{player_name} 下一步最可能采取什么行动？"

        options = [
            QAOption(key="A", text="Check（过牌）- 不下注，将行动权交给对手", is_correct=False),
            QAOption(key="B", text="Call（跟注）- 匹配对手的下注额", is_correct=False),
            QAOption(key="C", text="Raise（加注）- 提高下注额度", is_correct=False),
            QAOption(key="D", text="Fold（弃牌）- 放弃这手牌", is_correct=False),
        ]

        return {
            "question": question,
            "question_type": QuestionType.STRATEGY,
            "options": options,
            "option_count": 4,
        }

    @staticmethod
    def second_order_belief_question(
        player_a: str,
        player_b: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate second-order ToM question (3 options).
        "What does player A think player B believes about their hand?"
        """
        question = f"{player_a} 认为 {player_b} 对自己手牌实力的判断是什么？"

        options = [
            QAOption(
                key="A",
                text=f"{player_a} 认为 {player_b} 觉得自己手牌很强",
                is_correct=False,
            ),
            QAOption(
                key="B",
                text=f"{player_a} 认为 {player_b} 觉得自己手牌较弱",
                is_correct=False,
            ),
            QAOption(
                key="C",
                text=f"{player_a} 认为 {player_b} 对自己的手牌实力不确定",
                is_correct=False,
            ),
        ]

        return {
            "question": question,
            "question_type": QuestionType.SECOND_ORDER,
            "options": options,
            "option_count": 3,
        }

    @staticmethod
    def get_templates_for_action(action: ActionEvent) -> List[str]:
        """
        Get applicable template names for an action.

        Args:
            action: The action event

        Returns:
            List of template method names that apply
        """
        templates = []

        # Betting actions get intent and bluff questions
        if action.action_type in [ActionType.BET, ActionType.RAISE, ActionType.ALL_IN]:
            templates.extend(["intent_question", "binary_bluff_question"])

        # All actions can have strategy prediction (for opponent)
        if action.action_type != ActionType.FOLD:
            templates.append("strategy_prediction_question")

        # Significant bets get second-order questions
        if action.amount and action.amount > 0:
            templates.append("second_order_belief_question")

        return templates

