"""
ToM Belief Agent
Infers player beliefs about opponent's hand range
"""

import json
from typing import Any, Dict
from openai import OpenAI
from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import QAItem, AgentOutput
from .base import BaseReasoningAgent


@AgentRegistry.register_reasoning("tom_belief_agent")
class TomBeliefAgent(BaseReasoningAgent):
    """
    Infers what a player believes about their opponent's hand range.
    Uses LLM to reason about beliefs based on action history.
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
        Analyze player beliefs and their impact on option likelihoods.
        """
        # Build prompt for belief inference
        prompt = self._build_belief_prompt(question, perception_data)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            result = json.loads(response.choices[0].message.content)

            # Extract option scores
            option_scores = result.get("option_scores", {})
            belief_analysis = result.get("belief_analysis", "")
            confidence = result.get("confidence", 0.5)

            return AgentOutput(
                agent_name=self.name,
                timestamp=question.timestamp or 0,
                result={
                    "option_scores": option_scores,
                    "belief_analysis": belief_analysis,
                },
                confidence=confidence,
                reasoning_trace=belief_analysis,
            )

        except Exception as e:
            logger.error(f"Belief analysis failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                timestamp=question.timestamp or 0,
                result={"option_scores": {opt.key: 0.33 for opt in question.options}},
                confidence=0.0,
                reasoning_trace=f"Error: {e}",
            )

    def _build_belief_prompt(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
    ) -> str:
        """Build prompt for belief inference."""
        context = question.context

        action_desc = ""
        if context.action:
            action_desc = f"Player {context.action.get('player')} did {context.action.get('type')} ${context.action.get('amount', 0):,.0f}"

        action_history = ""
        if context.action_sequence:
            action_history = "\n".join([
                f"- {a.get('player')}: {a.get('type')} ${a.get('amount', 0):,.0f}"
                for a in context.action_sequence
            ])

        board_desc = ", ".join(context.board) if context.board else "No community cards yet"

        options_desc = "\n".join([
            f"{opt.key}) {opt.text}" for opt in question.options
        ])

        prompt = f"""You are analyzing a poker player's beliefs in a Theory of Mind reasoning task.

Current Situation:
- Phase: {context.phase.value if context.phase else 'Unknown'}
- Board: {board_desc}
- Pot: ${context.pot or 0:,.0f}
- Current Action: {action_desc}

Action History:
{action_history or 'No previous actions'}

Question: {question.question}

Options:
{options_desc}

Based on the betting patterns and game state, analyze what the acting player likely believes about their opponent's hand range. Then score each option based on how consistent it is with the player's likely beliefs.

Output JSON:
{{
    "belief_analysis": "Brief analysis of what the player likely believes",
    "option_scores": {{"A": 0.0-1.0, "B": 0.0-1.0, "C": 0.0-1.0}},
    "confidence": 0.0-1.0
}}

Note: Scores should sum to approximately 1.0."""

        return prompt

