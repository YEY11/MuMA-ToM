"""
ToM Social Goal Agent
Infers player's social/strategic goals (bluff/value/control)
"""

import json
from typing import Any, Dict
from openai import OpenAI
from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import QAItem, AgentOutput, SocialGoalType
from .base import BaseReasoningAgent


@AgentRegistry.register_reasoning("tom_social_agent")
class TomSocialAgent(BaseReasoningAgent):
    """
    Infers the player's strategic/social goal.
    Determines if they are bluffing, value betting, controlling, or trapping.
    """

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
        )
        self.model = config.LLM_MODEL_NAME

    def analyze(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
        **kwargs,
    ) -> AgentOutput:
        """
        Analyze strategic intent and score options accordingly.
        """
        # Build comprehensive prompt
        prompt = self._build_social_prompt(question, perception_data)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            result = json.loads(response.choices[0].message.content)

            option_scores = result.get("option_scores", {})
            social_goal = result.get("inferred_social_goal", "unknown")
            reasoning = result.get("reasoning", "")
            confidence = result.get("confidence", 0.5)

            return AgentOutput(
                agent_name=self.name,
                timestamp=question.timestamp or 0,
                result={
                    "option_scores": option_scores,
                    "inferred_social_goal": social_goal,
                    "reasoning": reasoning,
                },
                confidence=confidence,
                reasoning_trace=reasoning,
            )

        except Exception as e:
            logger.error(f"Social goal analysis failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                timestamp=question.timestamp or 0,
                result={"option_scores": {opt.key: 0.33 for opt in question.options}},
                confidence=0.0,
                reasoning_trace=f"Error: {e}",
            )

    def _build_social_prompt(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
    ) -> str:
        """Build prompt for social goal inference."""
        context = question.context

        action_desc = ""
        if context.action:
            action_desc = f"{context.action.get('player')} {context.action.get('type')} ${context.action.get('amount', 0):,.0f}"

        board_desc = ", ".join(context.board) if context.board else "Pre-flop (no board)"
        pot_desc = f"${context.pot:,.0f}" if context.pot else "Unknown"

        # Behavioral summary
        behavior_desc = "No behavioral data available"
        if context.behavioral_cues:
            cues = list(context.behavioral_cues.values())[0] if context.behavioral_cues else {}
            if isinstance(cues, dict):
                behavior_parts = []
                if cues.get("dominant_posture"):
                    behavior_parts.append(f"Posture: {cues['dominant_posture']}")
                if cues.get("dominant_emotion"):
                    behavior_parts.append(f"Emotion: {cues['dominant_emotion']}")
                if cues.get("fidgeting_detected"):
                    behavior_parts.append("Fidgeting detected")
                if behavior_parts:
                    behavior_desc = ", ".join(behavior_parts)

        decision_time_desc = ""
        if context.decision_time:
            decision_time_desc = f"Decision time: {context.decision_time:.1f} seconds"

        options_desc = "\n".join([
            f"{opt.key}) {opt.text}" for opt in question.options
        ])

        # Visible cards (audience mode only)
        cards_desc = ""
        if context.visible_cards:
            cards_desc = "\nKnown hole cards:\n"
            for player, cards in context.visible_cards.items():
                cards_desc += f"- {player}: {', '.join(cards)}\n"

        prompt = f"""You are an expert poker analyst determining a player's strategic intent.

Social Goal Categories:
- BLUFF: Player is representing a stronger hand than they have, trying to make opponent fold
- VALUE: Player believes they have the best hand and wants to extract value
- CONTROL: Player is managing pot size, not committing too much
- TRAP: Player is slowplaying a strong hand to induce action

Game State:
- Phase: {context.phase.value if context.phase else 'Unknown'}
- Board: {board_desc}
- Pot: {pot_desc}
- Action: {action_desc}
{cards_desc}

Behavioral Observations:
{behavior_desc}
{decision_time_desc}

Question: {question.question}

Options:
{options_desc}

Analyze the betting line, bet sizing relative to pot, and behavioral cues to determine the most likely social goal. Consider:
1. Is the bet size consistent with value or a bluff?
2. Does the betting line tell a coherent story?
3. Do behavioral cues match the stated action?

Output JSON:
{{
    "inferred_social_goal": "bluff|value|control|trap",
    "reasoning": "Brief explanation of your analysis",
    "option_scores": {{"A": 0.0-1.0, "B": 0.0-1.0, "C": 0.0-1.0}},
    "confidence": 0.0-1.0
}}

Scores should sum to approximately 1.0."""

        return prompt

