"""
Reasoning Pipeline
Orchestrates multi-agent ToM reasoning for answering questions
"""

from typing import Any, Dict, List

from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import EpisodeData, QADataset, QAItem, ReasoningResult

# Import agents package to trigger registration decorators
from LIMP_Poker_V3.reasoning import agents as _reasoning_agents  # noqa: F401


class ReasoningPipeline:
    """
    Multi-agent reasoning pipeline that:
    1. Coordinates multiple reasoning agents (pluggable)
    2. Aggregates their outputs into final predictions
    3. Provides explainable results with agent traces
    """

    def __init__(self):
        # Get enabled reasoning agents
        self.agents = AgentRegistry.get_reasoning_agents(config.AGENT_CONFIG)
        logger.info(
            f"Initialized ReasoningPipeline with agents: {[a.name for a in self.agents]}"
        )

        # Agent weights for aggregation (can be learned)
        self.agent_weights = {
            "PostureAgent": 0.2,
            "EquityAgent": 0.15,
            "TomBeliefAgent": 0.3,
            "TomSocialAgent": 0.35,
        }

    def answer_question(
        self,
        question: QAItem,
        perception_data: Dict[str, Any],
    ) -> ReasoningResult:
        """
        Answer a single question using all enabled agents.

        Args:
            question: The question to answer
            perception_data: Relevant perception data

        Returns:
            ReasoningResult with prediction and agent outputs
        """
        agent_outputs = []
        all_option_scores = {}

        # Collect outputs from all agents
        for agent in self.agents:
            try:
                output = agent.analyze(question, perception_data)
                agent_outputs.append(output)

                # Collect option scores
                scores = output.result.get("option_scores", {})
                weight = self.agent_weights.get(agent.name, 1.0 / len(self.agents))

                for opt_key, score in scores.items():
                    if opt_key not in all_option_scores:
                        all_option_scores[opt_key] = []
                    all_option_scores[opt_key].append((score, weight, agent.name))

                logger.debug(
                    f"{agent.name}: scores={scores}, confidence={output.confidence:.2f}"
                )

            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")

        # Aggregate scores
        final_scores = self._aggregate_scores(all_option_scores)

        # Determine prediction
        if final_scores:
            predicted_answer = max(final_scores, key=final_scores.get)
            confidence = final_scores[predicted_answer]
        else:
            predicted_answer = "A"  # Default
            confidence = 0.0

        return ReasoningResult(
            question_id=question.id,
            predicted_answer=predicted_answer,
            confidence=confidence,
            agent_outputs=agent_outputs,
            aggregation_method="weighted_sum",
            final_scores=final_scores,
        )

    def _aggregate_scores(
        self,
        all_option_scores: Dict[str, List[tuple]],
    ) -> Dict[str, float]:
        """
        Aggregate option scores from all agents.

        Uses weighted average based on agent weights and confidence.
        """
        final_scores = {}

        for opt_key, scores_list in all_option_scores.items():
            weighted_sum = 0.0
            weight_sum = 0.0

            for score, weight, agent_name in scores_list:
                weighted_sum += score * weight
                weight_sum += weight

            if weight_sum > 0:
                final_scores[opt_key] = weighted_sum / weight_sum

        # Normalize to sum to 1
        total = sum(final_scores.values())
        if total > 0:
            final_scores = {k: v / total for k, v in final_scores.items()}

        return final_scores

    def answer_dataset(
        self,
        dataset: QADataset,
        perception_data: Dict[str, Any],
    ) -> List[ReasoningResult]:
        """
        Answer all questions in a dataset.

        Args:
            dataset: QA dataset
            perception_data: Perception data for the episode

        Returns:
            List of ReasoningResults
        """
        results = []

        for question in dataset.questions:
            result = self.answer_question(question, perception_data)
            results.append(result)

            logger.info(
                f"Q: {question.id} | "
                f"Predicted: {result.predicted_answer} | "
                f"GT: {question.answer} | "
                f"Correct: {result.predicted_answer == question.answer}"
            )

        return results

    def evaluate(
        self,
        dataset: QADataset,
        results: List[ReasoningResult],
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.

        Args:
            dataset: QA dataset with ground truth
            results: Reasoning results

        Returns:
            Evaluation metrics
        """
        correct = 0
        total = len(results)

        by_type = {}
        by_level = {}

        for question, result in zip(dataset.questions, results):
            is_correct = result.predicted_answer == question.answer

            if is_correct:
                correct += 1

            # By question type
            q_type = question.question_type.value
            if q_type not in by_type:
                by_type[q_type] = {"correct": 0, "total": 0}
            by_type[q_type]["total"] += 1
            if is_correct:
                by_type[q_type]["correct"] += 1

            # By question level
            q_level = question.level.value
            if q_level not in by_level:
                by_level[q_level] = {"correct": 0, "total": 0}
            by_level[q_level]["total"] += 1
            if is_correct:
                by_level[q_level]["correct"] += 1

        # Calculate accuracies
        overall_acc = correct / total if total > 0 else 0

        type_acc = {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0
            for k, v in by_type.items()
        }

        level_acc = {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0
            for k, v in by_level.items()
        }

        return {
            "overall_accuracy": overall_acc,
            "total_questions": total,
            "correct_predictions": correct,
            "accuracy_by_type": type_acc,
            "accuracy_by_level": level_acc,
            "enabled_agents": [a.name for a in self.agents],
        }


def run_reasoning(
    episode_data: EpisodeData,
    dataset: QADataset,
) -> Dict[str, Any]:
    """
    Convenience function to run reasoning on a dataset.

    Args:
        episode_data: Perception output
        dataset: QA dataset

    Returns:
        Dict with results and evaluation
    """
    pipeline = ReasoningPipeline()

    # Convert episode data to perception dict
    perception_data = {
        "episode_id": episode_data.episode_id,
        "timeline": [p.model_dump() for p in episode_data.timeline],
        "meta": episode_data.meta,
    }

    # Answer all questions
    results = pipeline.answer_dataset(dataset, perception_data)

    # Evaluate
    evaluation = pipeline.evaluate(dataset, results)

    return {
        "results": [r.model_dump() for r in results],
        "evaluation": evaluation,
    }
