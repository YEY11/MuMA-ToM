"""
Evaluation Metrics
Metrics for evaluating reasoning performance
"""

from typing import List, Dict, Any
from LIMP_Poker_V3.core.schema import QADataset, ReasoningResult


class Metrics:
    """
    Evaluation metrics for the reasoning pipeline.
    """

    @staticmethod
    def accuracy(
        dataset: QADataset,
        results: List[ReasoningResult],
    ) -> float:
        """Calculate overall accuracy."""
        if not results:
            return 0.0

        correct = sum(
            1
            for q, r in zip(dataset.questions, results)
            if r.predicted_answer == q.answer
        )
        return correct / len(results)

    @staticmethod
    def accuracy_by_type(
        dataset: QADataset,
        results: List[ReasoningResult],
    ) -> Dict[str, float]:
        """Calculate accuracy by question type."""
        by_type = {}

        for q, r in zip(dataset.questions, results):
            q_type = q.question_type.value
            if q_type not in by_type:
                by_type[q_type] = {"correct": 0, "total": 0}

            by_type[q_type]["total"] += 1
            if r.predicted_answer == q.answer:
                by_type[q_type]["correct"] += 1

        return {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0
            for k, v in by_type.items()
        }

    @staticmethod
    def accuracy_by_level(
        dataset: QADataset,
        results: List[ReasoningResult],
    ) -> Dict[str, float]:
        """Calculate accuracy by question level."""
        by_level = {}

        for q, r in zip(dataset.questions, results):
            q_level = q.level.value
            if q_level not in by_level:
                by_level[q_level] = {"correct": 0, "total": 0}

            by_level[q_level]["total"] += 1
            if r.predicted_answer == q.answer:
                by_level[q_level]["correct"] += 1

        return {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0
            for k, v in by_level.items()
        }

    @staticmethod
    def confusion_matrix(
        dataset: QADataset,
        results: List[ReasoningResult],
    ) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix."""
        matrix = {}

        for q, r in zip(dataset.questions, results):
            gt = q.answer
            pred = r.predicted_answer

            if gt not in matrix:
                matrix[gt] = {}
            if pred not in matrix[gt]:
                matrix[gt][pred] = 0

            matrix[gt][pred] += 1

        return matrix

    @staticmethod
    def agent_contribution(
        results: List[ReasoningResult],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze contribution of each agent."""
        agent_stats = {}

        for result in results:
            for output in result.agent_outputs:
                name = output.agent_name
                if name not in agent_stats:
                    agent_stats[name] = {
                        "total_confidence": 0,
                        "count": 0,
                    }

                agent_stats[name]["total_confidence"] += output.confidence
                agent_stats[name]["count"] += 1

        # Calculate averages
        return {
            name: {
                "avg_confidence": stats["total_confidence"] / stats["count"]
                if stats["count"] > 0
                else 0,
                "invocations": stats["count"],
            }
            for name, stats in agent_stats.items()
        }

    @staticmethod
    def summary_report(
        dataset: QADataset,
        results: List[ReasoningResult],
    ) -> str:
        """Generate a text summary report."""
        overall = Metrics.accuracy(dataset, results)
        by_type = Metrics.accuracy_by_type(dataset, results)
        by_level = Metrics.accuracy_by_level(dataset, results)
        agent_contrib = Metrics.agent_contribution(results)

        lines = [
            "=" * 50,
            "Evaluation Report",
            "=" * 50,
            f"Total Questions: {len(results)}",
            f"Overall Accuracy: {overall:.2%}",
            "",
            "Accuracy by Question Type:",
        ]

        for q_type, acc in by_type.items():
            lines.append(f"  {q_type}: {acc:.2%}")

        lines.extend(["", "Accuracy by Question Level:"])
        for level, acc in by_level.items():
            lines.append(f"  {level}: {acc:.2%}")

        lines.extend(["", "Agent Contributions:"])
        for agent, stats in agent_contrib.items():
            lines.append(
                f"  {agent}: avg_conf={stats['avg_confidence']:.2f}, "
                f"invocations={stats['invocations']}"
            )

        lines.append("=" * 50)

        return "\n".join(lines)

