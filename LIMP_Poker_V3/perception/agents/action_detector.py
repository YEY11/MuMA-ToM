"""
Action Detection Agent
Detects player actions by analyzing state transitions
"""

from typing import List, Optional, Dict, Any
from loguru import logger

from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import (
    ActionEvent,
    ActionType,
    GameState,
    PlayerState,
    BehavioralCues,
)
from .base import BasePerceptionAgent


@AgentRegistry.register_perception("action_detector")
class ActionDetector(BasePerceptionAgent):
    """
    Rule-based agent for detecting actions from state transitions.
    Compares consecutive GameStates to infer player actions.
    """

    def __init__(self):
        super().__init__()
        # Threshold to ignore small stack fluctuations (noise)
        self.stack_change_threshold = 100

    def process(self, image_path: str, timestamp: float, **kwargs) -> Dict[str, Any]:
        """
        This agent doesn't process images directly.
        Use detect_actions() with GameState pairs instead.
        """
        return {"message": "Use detect_actions() method instead"}

    def detect_actions(
        self,
        prev_state: GameState,
        curr_state: GameState,
        interval_states: Optional[List[GameState]] = None,
    ) -> List[ActionEvent]:
        """
        Detect actions by comparing two consecutive game states.

        Args:
            prev_state: Previous game state
            curr_state: Current game state
            interval_states: Optional list of states between actions
                            (for behavioral sequence analysis)

        Returns:
            List of detected ActionEvents
        """
        actions = []

        for curr_p in curr_state.players:
            prev_p = self._find_player(prev_state, curr_p.name)
            if not prev_p:
                continue

            # Detect actions based on state changes
            action = self._detect_player_action(prev_p, curr_p, prev_state, curr_state)

            if action:
                # Enrich with behavioral analysis if interval states available
                if interval_states:
                    action = self._enrich_with_behavioral_data(
                        action, curr_p.name, interval_states, prev_state.timestamp
                    )

                actions.append(action)

        return actions

    def _find_player(
        self, state: GameState, name: str
    ) -> Optional[PlayerState]:
        """Find player by name in a game state"""
        for p in state.players:
            if p.name == name:
                return p
        return None

    def _detect_player_action(
        self,
        prev_p: PlayerState,
        curr_p: PlayerState,
        prev_state: GameState,
        curr_state: GameState,
    ) -> Optional[ActionEvent]:
        """
        Detect action for a single player.

        Returns:
            ActionEvent if action detected, None otherwise
        """
        # Handle None stacks gracefully
        prev_stack = prev_p.stack if prev_p.stack is not None else 0
        curr_stack = curr_p.stack if curr_p.stack is not None else 0

        # Skip if both stacks are None (can't determine action)
        if prev_p.stack is None and curr_p.stack is None:
            return None

        stack_diff = prev_stack - curr_stack

        # 1. Fold Detection: Player becomes inactive
        if prev_p.is_active and not curr_p.is_active:
            return ActionEvent(
                timestamp=curr_state.timestamp,
                player_name=curr_p.name,
                action_type=ActionType.FOLD,
                amount=0.0,
                decision_start_time=prev_state.timestamp,
                duration=curr_state.timestamp - prev_state.timestamp,
                visual_context={
                    "prev_state": {"is_active": prev_p.is_active},
                    "curr_state": {"is_active": curr_p.is_active},
                },
                detection_source="visual",
            )

        # 2. Betting Action: Stack decreased significantly
        if stack_diff > self.stack_change_threshold:
            # Determine action type based on pot and previous bets
            action_type = self._classify_bet_action(
                stack_diff, prev_state, curr_state
            )

            return ActionEvent(
                timestamp=curr_state.timestamp,
                player_name=curr_p.name,
                action_type=action_type,
                amount=stack_diff,
                decision_start_time=prev_state.timestamp,
                duration=curr_state.timestamp - prev_state.timestamp,
                visual_context={
                    "stack_before": prev_stack,
                    "stack_after": curr_stack,
                    "pot_before": prev_state.pot,
                    "pot_after": curr_state.pot,
                },
                detection_source="visual",
            )

        # 3. Check Detection: Phase unchanged, no betting, player still active
        # This is harder to detect without explicit visual indicators
        # For now, we don't explicitly detect checks

        return None

    def _classify_bet_action(
        self,
        amount: float,
        prev_state: GameState,
        curr_state: GameState,
    ) -> ActionType:
        """
        Classify betting action type (Bet/Call/Raise/All-in).

        This is a simplified heuristic. A full implementation would
        track betting history within the current betting round.
        """
        prev_pot = prev_state.pot or 0
        curr_pot = curr_state.pot or 0
        pot_increase = curr_pot - prev_pot

        # All-in detection: Stack went to near zero
        for p in curr_state.players:
            if p.stack is not None and p.stack < 100:
                return ActionType.ALL_IN

        # Simple heuristic based on pot ratio
        # In reality, need to track previous bet amounts
        if pot_increase > amount * 1.5:
            return ActionType.RAISE
        elif prev_pot > 0:
            return ActionType.CALL
        else:
            return ActionType.BET

    def _enrich_with_behavioral_data(
        self,
        action: ActionEvent,
        player_name: str,
        interval_states: List[GameState],
        start_time: float,
    ) -> ActionEvent:
        """
        Enrich action with behavioral sequence data from interval states.

        Args:
            action: The detected action
            player_name: Name of the acting player
            interval_states: States between previous and current action
            start_time: Start of decision interval

        Returns:
            Enriched ActionEvent
        """
        behavioral_sequence = []

        for state in interval_states:
            player = self._find_player(state, player_name)
            if player and player.behavioral_cues:
                behavioral_sequence.append(player.behavioral_cues)

        if behavioral_sequence:
            action.behavioral_sequence = behavioral_sequence
            action.decision_frame_count = len(behavioral_sequence)
            action.behavioral_summary = self._summarize_behavior(behavioral_sequence)

        return action

    def _summarize_behavior(
        self, sequence: List[BehavioralCues]
    ) -> Dict[str, Any]:
        """
        Summarize behavioral patterns over the decision interval.

        Args:
            sequence: List of BehavioralCues from interval frames

        Returns:
            Summary dict with behavioral insights
        """
        if not sequence:
            return {}

        postures = [c.posture for c in sequence if c.posture]
        hands = [c.hands for c in sequence if c.hands]
        gazes = [c.gaze for c in sequence if c.gaze]
        emotions = [c.facial_emotion.value if c.facial_emotion else None for c in sequence]
        emotions = [e for e in emotions if e]

        def most_common(lst):
            if not lst:
                return None
            return max(set(lst), key=lst.count)

        def has_change(lst):
            return len(set(lst)) > 1 if lst else False

        return {
            "dominant_posture": most_common(postures),
            "posture_changed": has_change(postures),
            "dominant_hands": most_common(hands),
            "fidgeting_detected": any(
                h in ["Playing with chips", "Touching face"] for h in hands
            ),
            "dominant_gaze": most_common(gazes),
            "gaze_changed": has_change(gazes),
            "dominant_emotion": most_common(emotions),
            "emotion_changed": has_change(emotions),
            "frame_count": len(sequence),
        }

