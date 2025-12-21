from typing import List
from LIMP_Poker_V2.core.schema import ActionEvent, ActionType, GameState
from LIMP_Poker_V2.perception.agents.board_agent import BoardParsingAgent


class ActionDetectionAgent:
    """
    Detects actions (Check, Bet, Raise, Fold, Call) by analyzing state transitions.
    """

    def __init__(self):
        self.board_agent = BoardParsingAgent()

    def detect_actions(
        self, prev_state: GameState, current_state: GameState, timestamp: float
    ) -> List[ActionEvent]:
        """
        Compare two consecutive game states to infer actions.
        This is a rule-based inference engine.
        """
        actions = []

        # 1. Identify active players (who acted?)
        # For now, we iterate all players and check stack/pot changes

        for i, curr_p in enumerate(current_state.players):
            # Find corresponding prev player
            prev_p = next(
                (p for p in prev_state.players if p.name == curr_p.name), None
            )
            if not prev_p:
                continue

            # Logic: Stack decrease + Pot increase = Bet/Call/Raise
            stack_diff = prev_p.stack - curr_p.stack

            # Threshold to ignore noise
            if stack_diff > 0:
                # Player put money in
                # Heuristic:
                # If pot increased by roughly stack_diff -> Bet/Call/Raise
                # (Ignoring rake or multi-way nuances for V1)

                action_type = ActionType.UNKNOWN
                # Simple logic: Need context of previous bet to distinguish Call vs Raise
                # For now, we label as "bet" broadly
                action_type = ActionType.BET

                actions.append(
                    ActionEvent(
                        timestamp=timestamp,
                        player_name=curr_p.name,
                        action_type=action_type,
                        amount=stack_diff,
                        visual_context={
                            "stack_before": prev_p.stack,
                            "stack_after": curr_p.stack,
                        },
                    )
                )

            # Logic: Fold detection
            # Harder to detect purely from stack. Usually visual cue "cards mucked".
            # Can inferred if player becomes "inactive" or cards disappear.
            # Assuming 'is_active' flag is updated by BoardAgent.
            if prev_p.is_active and not curr_p.is_active:
                actions.append(
                    ActionEvent(
                        timestamp=timestamp,
                        player_name=curr_p.name,
                        action_type=ActionType.FOLD,
                        amount=0.0,
                    )
                )

        # Logic: Check detection
        # If active player didn't put money in, and phase didn't change...
        # Hard to pinpoint timestamp without OCR of "Check" indicator.
        # We might need explicit "Action Indicator" detection in BoardAgent.

        return actions
