"""
Equity Calculation Agent
Calculates hand equity using eval7 or heuristics
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import QAItem, AgentOutput
from .base import BaseReasoningAgent


@AgentRegistry.register_reasoning("equity_agent")
class EquityAgent(BaseReasoningAgent):
    """
    Calculates hand equity (winning probability) using eval7 library.
    Can work with complete or partial information.
    """

    def __init__(self):
        super().__init__()
        self.eval7_available = self._check_eval7()

    def _check_eval7(self) -> bool:
        """Check if eval7 is available."""
        try:
            import eval7
            return True
        except ImportError:
            logger.warning("eval7 not available, using heuristic equity")
            return False

    def analyze(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
        **kwargs,
    ) -> AgentOutput:
        """
        Calculate equity for relevant players.
        """
        board = question.context.board or []
        visible_cards = question.context.visible_cards or {}

        equities = {}
        evidence = []

        if self.eval7_available and visible_cards:
            equities = self._calculate_eval7_equity(board, visible_cards)
            evidence.append(f"Calculated via eval7: {equities}")
        else:
            # Heuristic equity based on betting patterns
            equities = self._heuristic_equity(question, perception_data)
            evidence.append(f"Heuristic estimate: {equities}")

        # Map to option scores
        # Higher equity player is less likely to be bluffing
        option_scores = self._equity_to_scores(equities, question)

        return AgentOutput(
            agent_name=self.name,
            timestamp=question.timestamp or 0,
            result={
                "option_scores": option_scores,
                "equities": equities,
            },
            confidence=0.7 if self.eval7_available else 0.4,
            reasoning_trace="; ".join(evidence),
        )

    def _calculate_eval7_equity(
        self,
        board: List[str],
        hole_cards: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Calculate equity using eval7 Monte Carlo simulation.
        """
        try:
            import eval7

            # Parse cards
            board_cards = [eval7.Card(c) for c in board]
            
            players = list(hole_cards.keys())
            if len(players) != 2:
                return {}

            hands = []
            for player in players:
                cards = hole_cards.get(player, [])
                if len(cards) >= 2:
                    hands.append([eval7.Card(c) for c in cards[:2]])
                else:
                    return {}

            # Monte Carlo simulation
            wins = [0, 0]
            ties = 0
            n_simulations = 10000

            deck = eval7.Deck()
            # Remove known cards
            for card in board_cards:
                deck.cards.remove(card)
            for hand in hands:
                for card in hand:
                    deck.cards.remove(card)

            remaining_cards = 5 - len(board_cards)

            for _ in range(n_simulations):
                deck.shuffle()
                complete_board = board_cards + deck.peek(remaining_cards)
                
                scores = [
                    eval7.evaluate(hand + complete_board)
                    for hand in hands
                ]
                
                if scores[0] > scores[1]:
                    wins[0] += 1
                elif scores[1] > scores[0]:
                    wins[1] += 1
                else:
                    ties += 1

            total = n_simulations
            equities = {
                players[0]: (wins[0] + ties / 2) / total,
                players[1]: (wins[1] + ties / 2) / total,
            }

            return equities

        except Exception as e:
            logger.error(f"eval7 calculation failed: {e}")
            return {}

    def _heuristic_equity(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Estimate equity based on betting patterns (when cards unknown).
        """
        # Simple heuristic: aggressive betting suggests either strong hand or bluff
        # We return neutral estimates
        action = question.context.action
        if not action:
            return {}

        player = action.get("player", "Unknown")
        
        # Default to 50-50
        return {player: 0.5, "opponent": 0.5}

    def _equity_to_scores(
        self,
        equities: Dict[str, float],
        question: QAItem,
    ) -> Dict[str, float]:
        """
        Convert equity to option scores.
        
        Logic: Low equity + aggressive bet = more likely bluff
        """
        option_scores = {opt.key: 0.33 for opt in question.options}

        action = question.context.action
        if not action or not equities:
            return option_scores

        player = action.get("player", "")
        player_equity = equities.get(player, 0.5)

        # Low equity with big bet suggests bluff
        if player_equity < 0.35:
            option_scores["A"] = 0.5  # Bluff more likely
            option_scores["B"] = 0.2  # Value less likely
        elif player_equity > 0.65:
            option_scores["A"] = 0.2  # Bluff less likely
            option_scores["B"] = 0.5  # Value more likely
        else:
            option_scores["A"] = 0.33
            option_scores["B"] = 0.33

        option_scores["C"] = 1.0 - option_scores["A"] - option_scores["B"]

        return option_scores

