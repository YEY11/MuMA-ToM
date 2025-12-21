import argparse
import os

from loguru import logger

from LIMP_Poker_V2.perception.pipeline import PerceptionPipeline
from LIMP_Poker_V2.scripts.preprocess_video import preprocess_video


def main():
    parser = argparse.ArgumentParser(description="Run MuMA-ToM V2 Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file (mp4)")
    parser.add_argument(
        "--output_root",
        default="datasets/processed",
        help="Root directory for processed output",
    )
    parser.add_argument(
        "--episode_id", help="Optional episode ID (default: video filename)"
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip ffmpeg extraction if already done",
    )

    args = parser.parse_args()

    # Setup Paths
    if args.episode_id:
        ep_id = args.episode_id
    else:
        ep_id = os.path.splitext(os.path.basename(args.video))[0]

    episode_dir = os.path.join(args.output_root, ep_id)
    output_json = os.path.join(episode_dir, "perception_output.json")

    # 1. Preprocess
    if not args.skip_preprocess:
        logger.info(f"=== Step 1: Preprocessing {ep_id} ===")
        # Remove hardcoded fps=30 so it uses config.FPS (default 1)
        preprocess_video(args.video, episode_dir)
    else:
        logger.info(f"Skipping preprocessing for {ep_id}")

    # 2. Perception Pipeline
    logger.info("=== Step 2: Running Perception Pipeline ===")
    pipeline = PerceptionPipeline()
    pipeline.run(episode_dir, output_json)

    logger.info("=== Done ===")
    logger.info(f"Results saved to: {output_json}")


if __name__ == "__main__":
    main()
