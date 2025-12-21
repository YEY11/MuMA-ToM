"""
Base class for Reasoning Agents
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from LIMP_Poker_V3.core.schema import QAItem, AgentOutput


class BaseReasoningAgent(ABC):
    """
    Abstract base class for all reasoning layer agents.
    Reasoning agents analyze perception data and contribute to answering questions.
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def analyze(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
        **kwargs,
    ) -> AgentOutput:
        """
        Analyze perception data in context of a question.

        Args:
            question: The QA item being answered
            perception_data: Relevant perception data
            **kwargs: Additional arguments

        Returns:
            AgentOutput with analysis results
        """
        pass

    def get_option_scores(self, output: AgentOutput) -> Dict[str, float]:
        """
        Extract option scores from agent output.

        Args:
            output: Agent output

        Returns:
            Dict mapping option keys to scores
        """
        return output.result.get("option_scores", {})

    def __repr__(self):
        return f"<{self.name}>"

