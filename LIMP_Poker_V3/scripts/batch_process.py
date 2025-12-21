"""
Batch Processing Script
Process multiple videos through the pipeline
"""

import os
import glob
import json
import argparse
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from LIMP_Poker_V3.main import run_full_pipeline


def batch_process(
    video_dir: str,
    output_dir: str,
    pattern: str = "*.mp4",
    max_workers: int = 1,
    skip_existing: bool = True,
) -> dict:
    """
    Process multiple videos in batch.

    Args:
        video_dir: Directory containing videos
        output_dir: Output directory
        pattern: Glob pattern for video files
        max_workers: Number of parallel workers
        skip_existing: Skip already processed episodes

    Returns:
        Summary of processing results
    """
    # Find videos
    video_files = glob.glob(os.path.join(video_dir, pattern))
    logger.info(f"Found {len(video_files)} videos to process")

    results = {"processed": [], "skipped": [], "failed": []}

    def process_one(video_path):
        episode_id = os.path.splitext(os.path.basename(video_path))[0]
        episode_dir = os.path.join(output_dir, episode_id)

        # Check if already processed
        if skip_existing and os.path.exists(
            os.path.join(episode_dir, "reasoning_results.json")
        ):
            return {"status": "skipped", "episode_id": episode_id}

        try:
            result = run_full_pipeline(video_path, output_dir)
            return {"status": "success", "episode_id": episode_id, "result": result}
        except Exception as e:
            logger.error(f"Failed to process {episode_id}: {e}")
            return {"status": "failed", "episode_id": episode_id, "error": str(e)}

    # Process videos
    if max_workers == 1:
        # Sequential processing
        for video_path in video_files:
            result = process_one(video_path)
            if result["status"] == "success":
                results["processed"].append(result["episode_id"])
            elif result["status"] == "skipped":
                results["skipped"].append(result["episode_id"])
            else:
                results["failed"].append(result["episode_id"])
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one, vp): vp for vp in video_files
            }
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "success":
                    results["processed"].append(result["episode_id"])
                elif result["status"] == "skipped":
                    results["skipped"].append(result["episode_id"])
                else:
                    results["failed"].append(result["episode_id"])

    # Summary
    logger.info(
        f"Batch complete: {len(results['processed'])} processed, "
        f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process poker videos")
    parser.add_argument("--video-dir", required=True, help="Directory with videos")
    parser.add_argument("--output", default="datasets/processed_v3", help="Output dir")
    parser.add_argument("--pattern", default="*.mp4", help="Video file pattern")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Reprocess existing")

    args = parser.parse_args()

    results = batch_process(
        video_dir=args.video_dir,
        output_dir=args.output,
        pattern=args.pattern,
        max_workers=args.workers,
        skip_existing=not args.force,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

