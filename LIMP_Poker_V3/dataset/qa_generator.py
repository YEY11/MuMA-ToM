"""
QA Dataset Generator
Automatically generates QA pairs from perception and GT data
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.schema import (
    EpisodeData,
    PhaseData,
    ActionEvent,
    QAItem,
    QADataset,
    QAContext,
    QAOption,
    ToMLabels,
    QuestionLevel,
    QuestionType,
    SocialGoalType,
    ActionType,
)
from .templates.action_level import ActionLevelTemplates
from .templates.phase_level import PhaseLevelTemplates


class QAGenerator:
    """
    Generates QA dataset from perception results and ground truth.

    Supports:
    - Action-level questions (about specific actions)
    - Phase-level questions (about game phases)
    - Multiple question types with varying option counts
    """

    def __init__(self):
        self.action_templates = ActionLevelTemplates()
        self.phase_templates = PhaseLevelTemplates()
        self.protocol = config.PROTOCOL_MODE

    def generate(
        self,
        episode_data: EpisodeData,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> QADataset:
        """
        Generate QA dataset for an episode.

        Args:
            episode_data: Perception output
            ground_truth: Audio GT data (optional)

        Returns:
            QADataset with generated questions
        """
        questions = []
        episode_id = episode_data.episode_id

        logger.info(f"Generating QA for episode: {episode_id}")

        # Generate action-level questions
        if "action" in config.QA_LEVELS:
            action_qs = self._generate_action_questions(episode_data, ground_truth)
            questions.extend(action_qs)
            logger.info(f"Generated {len(action_qs)} action-level questions")

        # Generate phase-level questions
        if "phase" in config.QA_LEVELS:
            phase_qs = self._generate_phase_questions(episode_data, ground_truth)
            questions.extend(phase_qs)
            logger.info(f"Generated {len(phase_qs)} phase-level questions")

        return QADataset(
            episode_id=episode_id,
            protocol=self.protocol,
            questions=questions,
        )

    def _generate_action_questions(
        self,
        episode_data: EpisodeData,
        ground_truth: Optional[Dict[str, Any]],
    ) -> List[QAItem]:
        """Generate questions for individual actions."""
        questions = []
        action_count = 0

        for phase_data in episode_data.timeline:
            for action in phase_data.actions:
                # Skip folds and checks (less interesting)
                if action.action_type in [ActionType.FOLD, ActionType.CHECK]:
                    continue

                action_count += 1
                action_id = f"{episode_data.episode_id}_act_{action_count:03d}"

                # Build context
                context = self._build_action_context(action, phase_data, episode_data)

                # Get GT labels if available
                gt_labels = self._get_gt_for_action(action, ground_truth)

                # Generate intent question
                intent_q = self._create_intent_question(
                    action_id, action, context, gt_labels
                )
                if intent_q:
                    questions.append(intent_q)

                # Generate binary bluff question for significant bets
                if action.amount and action.amount > 10000:
                    bluff_q = self._create_bluff_question(
                        f"{action_id}_bluff", action, context, gt_labels
                    )
                    if bluff_q:
                        questions.append(bluff_q)

        return questions

    def _generate_phase_questions(
        self,
        episode_data: EpisodeData,
        ground_truth: Optional[Dict[str, Any]],
    ) -> List[QAItem]:
        """Generate questions for game phases."""
        questions = []

        for i, phase_data in enumerate(episode_data.timeline):
            phase_id = f"{episode_data.episode_id}_phase_{i:02d}"

            # Skip if no meaningful actions
            if not phase_data.actions:
                continue

            # Build phase context
            context = self._build_phase_context(phase_data, episode_data)

            # Get players from phase
            players = self._get_phase_players(phase_data)
            if len(players) < 2:
                continue

            # Strategy question for each player
            for player in players:
                strategy_q = self._create_phase_strategy_question(
                    f"{phase_id}_{player}", player, phase_data, context
                )
                if strategy_q:
                    questions.append(strategy_q)

            # Advantage question
            advantage_q = self._create_advantage_question(
                f"{phase_id}_advantage", players, phase_data, context
            )
            if advantage_q:
                questions.append(advantage_q)

        return questions

    def _build_action_context(
        self,
        action: ActionEvent,
        phase_data: PhaseData,
        episode_data: EpisodeData,
    ) -> QAContext:
        """Build context for action-level question."""
        # Get board and pot from phase
        board = phase_data.final_state.board if phase_data.final_state else []
        pot = phase_data.final_state.pot if phase_data.final_state else None

        # Get visible cards (audience protocol only)
        visible_cards = None
        if self.protocol == "audience":
            # Would be populated from GT
            visible_cards = {}

        return QAContext(
            phase=phase_data.phase,
            board=board,
            pot=pot,
            action={
                "player": action.player_name,
                "type": action.action_type.value,
                "amount": action.amount,
            },
            decision_time=action.duration,
            behavioral_cues=(
                {action.player_name: action.behavioral_summary}
                if action.behavioral_summary
                else None
            ),
            visible_cards=visible_cards,
        )

    def _build_phase_context(
        self,
        phase_data: PhaseData,
        episode_data: EpisodeData,
    ) -> QAContext:
        """Build context for phase-level question."""
        board = phase_data.final_state.board if phase_data.final_state else []
        pot = phase_data.final_state.pot if phase_data.final_state else None

        # Build action sequence
        action_sequence = [
            {
                "player": a.player_name,
                "type": a.action_type.value,
                "amount": a.amount,
            }
            for a in phase_data.actions
        ]

        # Aggregate behavioral cues from all actions in this phase
        aggregated_cues = {}
        total_decision_time = 0.0

        for action in phase_data.actions:
            # Aggregate behavioral summary per player
            if action.behavioral_summary:
                player_name = action.player_name
                if player_name not in aggregated_cues:
                    aggregated_cues[player_name] = action.behavioral_summary
                else:
                    # Merge behavioral data (keep the latest or most significant)
                    existing = aggregated_cues[player_name]
                    # Update with new data, preserving existing if new is None
                    for key, value in action.behavioral_summary.items():
                        if value is not None:
                            existing[key] = value

            # Sum up decision times
            if action.duration:
                total_decision_time += action.duration

        return QAContext(
            phase=phase_data.phase,
            board=board,
            pot=pot,
            action_sequence=action_sequence,
            behavioral_cues=aggregated_cues if aggregated_cues else None,
            decision_time=total_decision_time if total_decision_time > 0 else None,
        )

    def _get_phase_players(self, phase_data: PhaseData) -> List[str]:
        """Get unique player names from phase."""
        players = set()
        for action in phase_data.actions:
            players.add(action.player_name)
        if phase_data.initial_state:
            for p in phase_data.initial_state.players:
                if p.is_active:
                    players.add(p.name)
        return list(players)

    def _get_gt_for_action(
        self,
        action: ActionEvent,
        ground_truth: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Get ground truth labels for an action."""
        if not ground_truth:
            return None

        action_gt = ground_truth.get("action_gt", [])
        for gt in action_gt:
            start = gt.get("start", 0)
            end = gt.get("end", 0)
            if start <= action.timestamp <= end + 5:
                return gt.get("labels", {})

        return None

    def _create_intent_question(
        self,
        q_id: str,
        action: ActionEvent,
        context: QAContext,
        gt_labels: Optional[Dict[str, Any]],
    ) -> Optional[QAItem]:
        """Create intent judgment question."""
        template = self.action_templates.intent_question(
            player_name=action.player_name,
            action_type=action.action_type,
            amount=action.amount,
            context={},
        )

        options = template["options"]

        # Determine answer from GT
        answer = "A"  # Default
        social_goal = None

        if gt_labels:
            if gt_labels.get("is_bluff"):
                answer = "A"
                social_goal = SocialGoalType.BLUFF
            elif gt_labels.get("is_value"):
                answer = "B"
                social_goal = SocialGoalType.VALUE
            else:
                answer = "C"
                social_goal = SocialGoalType.CONTROL

        # Mark correct option
        for opt in options:
            opt.is_correct = opt.key == answer

        return QAItem(
            id=q_id,
            level=QuestionLevel.ACTION,
            question_type=QuestionType.INTENT,
            protocol=self.protocol,
            timestamp=action.timestamp,
            phase=context.phase,
            context=context,
            question=template["question"],
            options=options,
            answer=answer,
            answer_source="audio_commentary" if gt_labels else "rule_based",
            tom_labels=ToMLabels(social_goal=social_goal) if social_goal else None,
        )

    def _create_bluff_question(
        self,
        q_id: str,
        action: ActionEvent,
        context: QAContext,
        gt_labels: Optional[Dict[str, Any]],
    ) -> Optional[QAItem]:
        """Create binary bluff question."""
        template = self.action_templates.binary_bluff_question(
            player_name=action.player_name,
            action_type=action.action_type,
            amount=action.amount,
            context={},
        )

        options = template["options"]

        # Determine answer from GT
        answer = "B"  # Default: not bluff
        if gt_labels and gt_labels.get("is_bluff"):
            answer = "A"

        for opt in options:
            opt.is_correct = opt.key == answer

        return QAItem(
            id=q_id,
            level=QuestionLevel.ACTION,
            question_type=QuestionType.BINARY,
            protocol=self.protocol,
            timestamp=action.timestamp,
            phase=context.phase,
            context=context,
            question=template["question"],
            options=options,
            answer=answer,
            answer_source="audio_commentary" if gt_labels else "rule_based",
        )

    def _create_phase_strategy_question(
        self,
        q_id: str,
        player_name: str,
        phase_data: PhaseData,
        context: QAContext,
    ) -> Optional[QAItem]:
        """Create phase strategy question."""
        # Summarize actions
        player_actions = [
            a for a in phase_data.actions if a.player_name == player_name
        ]
        if not player_actions:
            return None

        action_summary = ", ".join(
            [f"{a.action_type.value}" for a in player_actions]
        )

        template = self.phase_templates.phase_strategy_question(
            player_name=player_name,
            phase=phase_data.phase,
            action_summary=action_summary,
            context={},
        )

        options = template["options"]

        # Infer answer based on action patterns (rule-based)
        answer = self._infer_strategy_from_actions(player_actions)
        answer_source = "rule_based"

        for opt in options:
            opt.is_correct = opt.key == answer

        return QAItem(
            id=q_id,
            level=QuestionLevel.PHASE,
            question_type=QuestionType.INTENT,
            protocol=self.protocol,
            phase=phase_data.phase,
            context=context,
            question=template["question"],
            options=options,
            answer=answer,
            answer_source=answer_source,
        )

    def _infer_strategy_from_actions(
        self,
        actions: List[ActionEvent],
    ) -> str:
        """
        Infer strategy type based on action patterns.
        
        Returns:
            A/B/C corresponding to Aggressive/Conservative/Deceptive
        """
        if not actions:
            return "B"  # Default to conservative
        
        # Count action types
        raises = sum(1 for a in actions if a.action_type == ActionType.RAISE)
        bets = sum(1 for a in actions if a.action_type == ActionType.BET)
        calls = sum(1 for a in actions if a.action_type == ActionType.CALL)
        checks = sum(1 for a in actions if a.action_type == ActionType.CHECK)
        folds = sum(1 for a in actions if a.action_type == ActionType.FOLD)
        
        total_aggressive = raises + bets
        total_passive = calls + checks
        
        # Check behavioral cues for deception hints
        has_deception_hints = False
        for action in actions:
            if action.behavioral_summary:
                summary = action.behavioral_summary
                # Fidgeting, posture change, or emotion change might indicate deception
                if summary.get("fidgeting_detected") or summary.get("posture_changed"):
                    has_deception_hints = True
                    break
        
        # Infer strategy
        if total_aggressive > total_passive:
            return "A"  # Aggressive
        elif has_deception_hints and total_aggressive > 0:
            return "C"  # Deceptive (showing aggression with nervous tells)
        elif folds > 0:
            return "B"  # Conservative (folded)
        elif total_passive > total_aggressive:
            return "B"  # Conservative (mostly calling/checking)
        else:
            return "C"  # Deceptive (balanced or unclear)

    def _create_advantage_question(
        self,
        q_id: str,
        players: List[str],
        phase_data: PhaseData,
        context: QAContext,
    ) -> Optional[QAItem]:
        """Create phase advantage question."""
        if len(players) < 2:
            return None

        template = self.phase_templates.phase_winner_prediction_question(
            player_a=players[0],
            player_b=players[1],
            phase=phase_data.phase,
            context={},
        )

        options = template["options"]

        # Infer answer based on available information
        answer = self._infer_advantage(players, phase_data)
        
        for opt in options:
            opt.is_correct = opt.key == answer

        return QAItem(
            id=q_id,
            level=QuestionLevel.PHASE,
            question_type=QuestionType.INTENT,
            protocol=self.protocol,
            phase=phase_data.phase,
            context=context,
            question=template["question"],
            options=options,
            answer=answer,
            answer_source="rule_based",
        )

    def _infer_advantage(
        self,
        players: List[str],
        phase_data: PhaseData,
    ) -> str:
        """
        Infer which player has advantage based on action patterns and aggression.
        
        Returns:
            A = player_a advantage
            B = player_b advantage  
            C = unclear/even
        """
        if len(players) < 2:
            return "C"
        
        player_a, player_b = players[0], players[1]
        
        # Count aggressive actions per player
        a_aggression = 0
        b_aggression = 0
        
        for action in phase_data.actions:
            is_aggressive = action.action_type in [ActionType.RAISE, ActionType.BET]
            if action.player_name == player_a and is_aggressive:
                a_aggression += 1
            elif action.player_name == player_b and is_aggressive:
                b_aggression += 1
        
        # Check stack sizes if available
        a_stack = None
        b_stack = None
        if phase_data.final_state:
            for p in phase_data.final_state.players:
                if p.name == player_a:
                    a_stack = p.stack
                elif p.name == player_b:
                    b_stack = p.stack
        
        # Determine advantage
        if a_aggression > b_aggression:
            return "A"  # Player A more aggressive = likely has advantage
        elif b_aggression > a_aggression:
            return "B"  # Player B more aggressive
        elif a_stack and b_stack:
            # If same aggression, check stack sizes
            if a_stack > b_stack * 1.2:
                return "A"
            elif b_stack > a_stack * 1.2:
                return "B"
        
        return "C"  # Unclear

    def save(self, dataset: QADataset, output_path: str):
        """Save QA dataset to file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dataset.model_dump_json(indent=2, ensure_ascii=False))
        logger.info(f"QA dataset saved to {output_path}")

