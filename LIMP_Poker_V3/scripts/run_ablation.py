"""
Ablation Experiment Script
Run experiments with different agent configurations
"""

import os
import json
import argparse
from typing import Dict, List
from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.schema import QADataset, EpisodeData
from LIMP_Poker_V3.reasoning.pipeline import ReasoningPipeline
from LIMP_Poker_V3.evaluation.metrics import Metrics


# Predefined ablation configurations
ABLATION_CONFIGS = {
    "full": {
        "posture_agent": True,
        "equity_agent": True,
        "tom_belief_agent": True,
        "tom_social_agent": True,
    },
    "no_posture": {
        "posture_agent": False,
        "equity_agent": True,
        "tom_belief_agent": True,
        "tom_social_agent": True,
    },
    "no_equity": {
        "posture_agent": True,
        "equity_agent": False,
        "tom_belief_agent": True,
        "tom_social_agent": True,
    },
    "no_tom_belief": {
        "posture_agent": True,
        "equity_agent": True,
        "tom_belief_agent": False,
        "tom_social_agent": True,
    },
    "no_tom_social": {
        "posture_agent": True,
        "equity_agent": True,
        "tom_belief_agent": True,
        "tom_social_agent": False,
    },
    "no_tom": {
        "posture_agent": True,
        "equity_agent": True,
        "tom_belief_agent": False,
        "tom_social_agent": False,
    },
    "baseline_only": {
        "posture_agent": False,
        "equity_agent": False,
        "tom_belief_agent": False,
        "tom_social_agent": False,
    },
    "posture_only": {
        "posture_agent": True,
        "equity_agent": False,
        "tom_belief_agent": False,
        "tom_social_agent": False,
    },
    "tom_only": {
        "posture_agent": False,
        "equity_agent": False,
        "tom_belief_agent": True,
        "tom_social_agent": True,
    },
}


def run_ablation(
    episode_dir: str,
    configs: List[str] = None,
    output_path: str = None,
) -> Dict[str, Dict]:
    """
    Run ablation experiments on a processed episode.

    Args:
        episode_dir: Directory with perception and QA data
        configs: List of config names to run (default: all)
        output_path: Path to save results

    Returns:
        Dict mapping config name to evaluation results
    """
    # Load data
    perception_path = os.path.join(episode_dir, "perception_output.json")
    qa_path = os.path.join(episode_dir, "qa_dataset.json")

    with open(perception_path, "r") as f:
        episode_data = EpisodeData(**json.load(f))

    with open(qa_path, "r") as f:
        qa_dataset = QADataset(**json.load(f))

    perception_dict = {
        "episode_id": episode_data.episode_id,
        "timeline": [p.model_dump() for p in episode_data.timeline],
    }

    # Select configs
    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())

    results = {}

    for config_name in configs:
        if config_name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown config: {config_name}")
            continue

        logger.info(f"Running ablation: {config_name}")

        # Update config
        agent_config = ABLATION_CONFIGS[config_name].copy()
        # Keep perception agents enabled
        agent_config["board_agent"] = True
        agent_config["action_detector"] = True

        # Temporarily update global config
        original_config = config.AGENT_CONFIG.copy()
        config.AGENT_CONFIG.update(agent_config)

        try:
            # Re-initialize pipeline with new config
            # Note: This requires reimporting to pick up new config
            from LIMP_Poker_V3.core.registry import AgentRegistry

            pipeline = ReasoningPipeline()

            # Run reasoning
            reasoning_results = pipeline.answer_dataset(qa_dataset, perception_dict)

            # Evaluate
            evaluation = pipeline.evaluate(qa_dataset, reasoning_results)
            evaluation["config"] = agent_config

            results[config_name] = evaluation

            logger.info(
                f"  {config_name}: accuracy = {evaluation['overall_accuracy']:.2%}"
            )

        except Exception as e:
            logger.error(f"  {config_name} failed: {e}")
            results[config_name] = {"error": str(e)}

        finally:
            # Restore config
            config.AGENT_CONFIG = original_config

    # Print summary
    print("\n" + "=" * 60)
    print("Ablation Study Results")
    print("=" * 60)
    for config_name, eval_result in results.items():
        if "error" in eval_result:
            print(f"{config_name}: ERROR - {eval_result['error']}")
        else:
            acc = eval_result.get("overall_accuracy", 0)
            print(f"{config_name}: {acc:.2%}")
    print("=" * 60)

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--episode-dir",
        required=True,
        help="Directory with processed episode",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config names to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations",
    )

    args = parser.parse_args()

    if args.list_configs:
        print("Available ablation configurations:")
        for name, cfg in ABLATION_CONFIGS.items():
            enabled = [k for k, v in cfg.items() if v]
            print(f"  {name}: {enabled}")
        return

    results = run_ablation(
        episode_dir=args.episode_dir,
        configs=args.configs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

