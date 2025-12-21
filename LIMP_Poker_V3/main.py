"""
LIMP_Poker_V3 Main Entry Point
Unified interface for the complete pipeline
"""

import os
import json
import argparse
from loguru import logger

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.preprocessing.video_preprocessor import preprocess_video
from LIMP_Poker_V3.perception.pipeline import PerceptionPipeline
from LIMP_Poker_V3.annotation.audio_gt_agent import AudioGTAgent
from LIMP_Poker_V3.dataset.qa_generator import QAGenerator
from LIMP_Poker_V3.reasoning.pipeline import ReasoningPipeline
from LIMP_Poker_V3.evaluation.metrics import Metrics


def run_full_pipeline(
    video_path: str,
    output_dir: str,
    skip_preprocess: bool = False,
    skip_perception: bool = False,
    skip_qa_gen: bool = False,
    skip_reasoning: bool = False,
) -> dict:
    """
    Run the complete pipeline: preprocess -> perceive -> generate QA -> reason

    Args:
        video_path: Path to input video
        output_dir: Root output directory
        skip_preprocess: Skip video preprocessing
        skip_perception: Skip perception (use cached)
        skip_qa_gen: Skip QA generation (use cached)
        skip_reasoning: Skip reasoning evaluation

    Returns:
        Dict with all outputs and evaluation
    """
    # Setup paths
    episode_id = os.path.splitext(os.path.basename(video_path))[0]
    episode_dir = os.path.join(output_dir, episode_id)
    os.makedirs(episode_dir, exist_ok=True)

    perception_path = os.path.join(episode_dir, "perception_output.json")
    gt_path = os.path.join(episode_dir, "ground_truth.json")
    qa_path = os.path.join(episode_dir, "qa_dataset.json")
    results_path = os.path.join(episode_dir, "reasoning_results.json")

    logger.info(f"Processing episode: {episode_id}")
    logger.info(f"Protocol mode: {config.PROTOCOL_MODE}")
    config.print_config()

    results = {"episode_id": episode_id}

    # Step 1: Preprocess video
    if not skip_preprocess:
        logger.info("=== Step 1: Preprocessing Video ===")
        preprocess_result = preprocess_video(video_path, episode_dir)
        results["preprocess"] = preprocess_result
        if not preprocess_result.get("success"):
            logger.error("Preprocessing failed")
            return results
    else:
        logger.info("Skipping preprocessing")

    # Step 2: Perception
    if not skip_perception:
        logger.info("=== Step 2: Running Perception ===")
        perception_pipeline = PerceptionPipeline()
        episode_data = perception_pipeline.run(
            episode_dir, perception_path, use_cache=True
        )
        results["perception"] = {
            "phases": len(episode_data.timeline),
            "total_actions": sum(
                len(p.actions) for p in episode_data.timeline
            ),
        }
    else:
        logger.info("Loading cached perception data")
        from LIMP_Poker_V3.core.schema import EpisodeData
        with open(perception_path, "r") as f:
            episode_data = EpisodeData(**json.load(f))

    # Step 3: Ground Truth Extraction (Audio)
    audio_path = os.path.join(episode_dir, "audio.wav")
    gt_data = {}
    if os.path.exists(audio_path):
        logger.info("=== Step 3: Extracting Ground Truth ===")
        gt_agent = AudioGTAgent()
        gt_data = gt_agent.process(audio_path)
        with open(gt_path, "w") as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)
        results["ground_truth"] = {
            "has_transcript": bool(gt_data.get("transcript")),
            "action_gt_count": len(gt_data.get("action_gt", [])),
        }

    # Step 4: QA Generation
    if not skip_qa_gen:
        logger.info("=== Step 4: Generating QA Dataset ===")
        qa_generator = QAGenerator()
        qa_dataset = qa_generator.generate(episode_data, gt_data)
        qa_generator.save(qa_dataset, qa_path)
        results["qa_dataset"] = {
            "total_questions": len(qa_dataset.questions),
            "action_level": len(qa_dataset.get_by_level("action")),
            "phase_level": len(qa_dataset.get_by_level("phase")),
        }
    else:
        logger.info("Loading cached QA dataset")
        from LIMP_Poker_V3.core.schema import QADataset
        with open(qa_path, "r") as f:
            qa_dataset = QADataset(**json.load(f))

    # Step 5: Reasoning
    if not skip_reasoning:
        logger.info("=== Step 5: Running Reasoning ===")
        reasoning_pipeline = ReasoningPipeline()

        perception_dict = {
            "episode_id": episode_data.episode_id,
            "timeline": [p.model_dump() for p in episode_data.timeline],
        }

        reasoning_results = reasoning_pipeline.answer_dataset(
            qa_dataset, perception_dict
        )

        # Evaluate
        evaluation = reasoning_pipeline.evaluate(qa_dataset, reasoning_results)
        results["evaluation"] = evaluation

        # Generate report
        report = Metrics.summary_report(qa_dataset, reasoning_results)
        print(report)

        # Save results
        with open(results_path, "w") as f:
            json.dump(
                {
                    "evaluation": evaluation,
                    "results": [r.model_dump() for r in reasoning_results],
                },
                f,
                indent=2,
            )

    logger.info("=== Pipeline Complete ===")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="LIMP_Poker_V3: Multi-modal Multi-Agent ToM Reasoning"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        default="datasets/processed_v3",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip video preprocessing",
    )
    parser.add_argument(
        "--skip-perception",
        action="store_true",
        help="Skip perception (use cached)",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA generation (use cached)",
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        help="Skip reasoning evaluation",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print configuration and exit",
    )

    args = parser.parse_args()

    if args.print_config:
        config.print_config()
        return

    results = run_full_pipeline(
        video_path=args.video,
        output_dir=args.output,
        skip_preprocess=args.skip_preprocess,
        skip_perception=args.skip_perception,
        skip_qa_gen=args.skip_qa,
        skip_reasoning=args.skip_reasoning,
    )

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

