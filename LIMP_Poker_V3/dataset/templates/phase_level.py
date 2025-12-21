"""
Phase-Level Question Templates
Templates for generating questions about game phases
"""

from typing import Dict, List, Any
from LIMP_Poker_V3.core.schema import (
    PhaseData,
    PhaseType,
    QuestionType,
    QAOption,
)


class PhaseLevelTemplates:
    """
    Templates for phase-level questions.
    Questions about overall strategy during a phase.
    """

    @staticmethod
    def phase_strategy_question(
        player_name: str,
        phase: PhaseType,
        action_summary: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate phase strategy question (3 options).
        "What was the player's overall strategy during this phase?"
        """
        phase_name = phase.value

        question = f"在 {phase_name} 阶段，{player_name} 的整体策略最可能是什么？"

        options = [
            QAOption(
                key="A",
                text=f"激进诈唬策略 - {player_name} 利用位置或牌面优势施加压力，试图逼对手弃牌",
                is_correct=False,
            ),
            QAOption(
                key="B",
                text=f"价值导向策略 - {player_name} 认为自己领先，逐步建立底池以最大化收益",
                is_correct=False,
            ),
            QAOption(
                key="C",
                text=f"控池防守策略 - {player_name} 保持谨慎，控制底池大小以保持灵活性",
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
    def phase_winner_prediction_question(
        player_a: str,
        player_b: str,
        phase: PhaseType,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate phase-end advantage question (3 options).
        "Who has the advantage at the end of this phase?"
        """
        phase_name = phase.value

        question = f"在 {phase_name} 阶段结束时，哪位玩家的处境更有利？"

        options = [
            QAOption(
                key="A",
                text=f"{player_a} - 基于牌面和行动，{player_a} 更可能占据优势",
                is_correct=False,
            ),
            QAOption(
                key="B",
                text=f"{player_b} - 基于牌面和行动，{player_b} 更可能占据优势",
                is_correct=False,
            ),
            QAOption(
                key="C",
                text="势均力敌 - 双方优势接近，局势尚不明朗",
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
    def phase_behavioral_insight_question(
        player_name: str,
        phase: PhaseType,
        behavioral_summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate behavioral insight question (3 options).
        "What does the player's behavior suggest about their hand?"
        """
        phase_name = phase.value

        question = f"根据 {player_name} 在 {phase_name} 阶段的行为表现，他的手牌实力最可能是？"

        options = [
            QAOption(
                key="A",
                text=f"强牌 - {player_name} 的行为表现出自信和放松",
                is_correct=False,
            ),
            QAOption(
                key="B",
                text=f"中等牌力 - {player_name} 的行为相对中性，难以判断",
                is_correct=False,
            ),
            QAOption(
                key="C",
                text=f"弱牌或诈唬 - {player_name} 的行为显示出紧张或刻意表演",
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
    def phase_decision_analysis_question(
        player_name: str,
        phase: PhaseType,
        key_action: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate key decision analysis question (2 options).
        "Was this a good decision?"
        """
        phase_name = phase.value

        question = f"考虑到已知信息，{player_name} 在 {phase_name} 阶段的 {key_action} 是否是正确的决策？"

        options = [
            QAOption(
                key="A",
                text=f"是正确决策 - 基于当时的信息，{player_name} 的选择是合理的",
                is_correct=False,
            ),
            QAOption(
                key="B",
                text=f"不是最优决策 - {player_name} 有更好的选择但没有采取",
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
    def get_templates_for_phase(phase_data: PhaseData) -> List[str]:
        """
        Get applicable template names for a phase.

        Args:
            phase_data: The phase data

        Returns:
            List of template method names that apply
        """
        templates = ["phase_strategy_question"]

        # If there are actions, add more question types
        if phase_data.actions:
            templates.append("phase_winner_prediction_question")

            # If behavioral data exists
            for action in phase_data.actions:
                if action.behavioral_summary:
                    templates.append("phase_behavioral_insight_question")
                    break

            # Key decisions in later phases
            if phase_data.phase in [PhaseType.TURN, PhaseType.RIVER]:
                templates.append("phase_decision_analysis_question")

        return templates

