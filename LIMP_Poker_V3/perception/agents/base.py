"""
Base class for Perception Agents
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from LIMP_Poker_V3.core.schema import GameState


class BasePerceptionAgent(ABC):
    """
    Abstract base class for all perception layer agents.
    Perception agents extract structured information from raw visual data.
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def process(self, image_path: str, timestamp: float, **kwargs) -> Dict[str, Any]:
        """
        Process a single frame and extract information.

        Args:
            image_path: Path to the frame image
            timestamp: Timestamp of the frame in seconds
            **kwargs: Additional arguments

        Returns:
            Dict containing extracted information
        """
        pass

    def __repr__(self):
        return f"<{self.name}>"

